import io
import math
import warnings
import numpy as np
import cv2
from PIL import Image, ImageOps

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_SIDE = 1600


class WipingBytesIO(io.BytesIO):
    """Zero the owned encoded-image buffer before any caller closes it."""
    def close(self):
        if self.closed:
            return
        view = self.getbuffer()
        try:
            view[:] = b'\0' * len(view)
        finally:
            view.release()
        super().close()


def wipe(value):
    """Best effort only: Python/PIL/CUDA can make immutable or hidden copies."""
    if isinstance(value, bytearray):
        value[:] = b'\0' * len(value)
    elif isinstance(value, np.ndarray) and value.flags.writeable:
        value.fill(0)
    elif isinstance(value, Image.Image):
        try:
            # Inspect Pillow's stored core directly instead of the public `im`
            # property. Newer Pillow versions can raise when `im` is read after
            # close; touching a lazy rejected image can also trigger decoding.
            core = value.__dict__.get('_im')
            is_live_core = core is not None and core.__class__.__name__ != 'DeferredError'
            if is_live_core:
                try:
                    value.paste(0, (0, 0, value.width, value.height))
                except (ValueError, AssertionError):
                    # Pillow may already have destroyed the pixel core.
                    pass
        finally:
            try:
                value.close()
            except (ValueError, AssertionError):
                pass
    elif isinstance(value, io.BytesIO):
        if value.closed:
            return
        view = value.getbuffer()
        try:
            view[:] = b'\0' * len(view)
        finally:
            view.release()
            value.close()


def decode_clean(data):
    """Apply EXIF orientation, then create a fresh RGB image with no metadata."""
    if not isinstance(data, (bytes, bytearray)) or not 0 < len(data) <= MAX_FILE_BYTES:
        raise ValueError('invalid_image_bytes')
    stream = WipingBytesIO(data)
    raw = oriented = rgb = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            raw = Image.open(stream)
            if raw.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('unsupported_image_format')
            if raw.width * raw.height > MAX_PIXELS or getattr(raw, 'n_frames', 1) != 1:
                raise ValueError('image_limits_exceeded')
            raw.load()
            oriented = ImageOps.exif_transpose(raw)
            rgb = oriented.convert('RGB')
            rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            clean = Image.new('RGB', rgb.size)
            clean.paste(rgb)
            clean.info.clear()
            return clean
    finally:
        # The image is fully loaded before this point; erase encoded bytes first.
        # PNG Image.close() can close caller-owned streams, so close is idempotent.
        try:
            wipe(stream)
        finally:
            for image in (rgb, oriented, raw):
                if image is not None:
                    wipe(image)


def encode_png(image):
    """No EXIF, GPS, ICC, text chunks, filenames or source metadata."""
    fresh = Image.new('RGB', image.size)
    fresh.paste(image)
    stream = io.BytesIO()
    try:
        fresh.save(stream, format='PNG', compress_level=3)
        return bytearray(stream.getvalue())
    finally:
        wipe(fresh)
        wipe(stream)


def lab_recolor(rgb, mask, target_rgb, strength=0.85):
    """Keep L* exactly in LAB before sRGB conversion; change chroma only.
    Dark-to-blond lightening is deliberately outside this baseline.
    Clipping/quantization can slightly change measured output luminance.
    Pixels outside the nonzero mask are bit-for-bit identical.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('invalid_rgb')
    if mask.shape != rgb.shape[:2] or not np.isfinite(mask).all():
        raise ValueError('invalid_mask')
    if not 0 <= strength <= 1:
        raise ValueError('invalid_strength')
    target = np.asarray(target_rgb, dtype=np.float32)
    if target.shape != (3,) or not np.isfinite(target).all() or (target < 0).any() or (target > 255).any():
        raise ValueError('invalid_target_color')
    lab = changed = rendered = delta = None
    try:
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor((target / 255).reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]
        selection = mask >= 0.75
        if selection.sum() < 32:
            raise ValueError('insufficient_hair_pixels')
        source_ab = np.median(lab[selection, 1:3], axis=0)
        delta = target_lab[1:3] - source_ab
        changed = lab.copy()
        changed[..., 1:3] += delta * (np.clip(mask, 0, 1) * strength)[..., None]
        rendered = np.rint(np.clip(cv2.cvtColor(changed, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)
        rendered[mask <= 0] = rgb[mask <= 0]
        result = rendered.copy()
        return result
    finally:
        for x in (lab, changed, rendered, delta):
            if x is not None:
                wipe(x)


def comparison_frame(original, aligned, reference, final):
    # Original and aligned are both shown: alignment must never hide identity changes.
    panels = [('Original', original), ('Aligned target', aligned),
              ('Reference', reference), ('Result (aligned crop)', final)]
    from PIL import ImageDraw
    frame = Image.new('RGB', (4 * 384, 416), 'white')
    draw = ImageDraw.Draw(frame)
    for i, (title, image) in enumerate(panels):
        thumb = ImageOps.contain(image, (384, 384), Image.Resampling.LANCZOS)
        frame.paste(thumb, (i * 384 + (384 - thumb.width) // 2, 32 + (384 - thumb.height) // 2))
        draw.text((i * 384 + 8, 9), title, fill='black')
        wipe(thumb)
    return frame
