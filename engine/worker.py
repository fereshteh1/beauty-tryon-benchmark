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


from hair_engine import HairEngine
from color_engine import ColorEngine
import os, sys, gc, time, json, struct, pickle, threading, contextlib
from pathlib import Path

GUARD_ACTIVE = False


def audit_guard(event, args):
    if not GUARD_ACTIVE:
        return
    if event == 'open':
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & write_flags):
            raise PermissionError('disk_write_blocked_during_request')
    if event in {'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.link', 'os.symlink',
                 'subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind', 'socket.getaddrinfo'}:
        raise PermissionError('side_effect_blocked_during_request')


@contextlib.contextmanager
def request_guard():
    global GUARD_ACTIVE
    before = GUARD_ACTIVE
    GUARD_ACTIVE = True
    try:
        yield
    finally:
        GUARD_ACTIVE = before


class MemorySampler:
    """NVML samples process/device usage; PyTorch reports exact allocator peaks.
    NVML samples may miss spikes <10 ms; allocator peaks do not.
    """
    def __init__(self):
        self.process_peak = None
        self.device_peak = None
        self.stop_event = threading.Event()
        try:
            import pynvml
            self.nvml = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nvml = None

    def sample(self):
        if self.nvml is None:
            return
        try:
            total = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20
            self.device_peak = max(self.device_peak or 0, total)
            procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            used = sum(p.usedGpuMemory for p in procs if p.pid == os.getpid()
                       and isinstance(p.usedGpuMemory, int) and p.usedGpuMemory < 2**60)
            if used:
                self.process_peak = max(self.process_peak or 0, used / 2**20)
        except Exception:
            pass

    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(.01):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.sample()
        self.stop_event.set()
        self.thread.join(timeout=1)






def evaluate(engine, request, provenance):
    import torch
    face = reference = result = aligned = aligned_ref = frame = None
    measurements = []
    output = {}
    started = time.perf_counter()
    try:
        face = decode_clean(request['face'])
        if request.get('reference'):
            reference = decode_clean(request['reference'])
        if isinstance(engine, HairEngine) and reference is None:
            raise ValueError('hairstyle_reference_required')
        decode_s = time.perf_counter() - started
        repetitions = int(request.get('repeats', 3))
        if not 1 <= repetitions <= 10:
            raise ValueError('invalid_repeats')
        for _ in range(repetitions):
            for image in (result, aligned, aligned_ref):
                if image is not None:
                    wipe(image)
            result = aligned = aligned_ref = None
            sampler = MemorySampler()
            sampler.start()
            torch.cuda.synchronize()
            baseline_alloc = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                with torch.inference_mode():
                    if isinstance(engine, ColorEngine):
                        result, aligned, aligned_ref, m = engine.run(
                            face, reference, request.get('target_rgb', (116, 52, 40)),
                            float(request.get('strength', .85)), bool(request.get('reference_color', False)))
                    else:
                        result, aligned, aligned_ref, m = engine.run(face, reference)
                torch.cuda.synchronize()
                # Includes CPU detection/alignment and output tensor->PIL conversion.
                m['pipeline_s'] = time.perf_counter() - t0
                m['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
                m['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
                m['incremental_allocated_mib'] = (torch.cuda.max_memory_allocated() - baseline_alloc) / 2**20
            finally:
                sampler.stop()
            m['nvml_process_peak_mib'] = sampler.process_peak
            m['nvml_device_peak_mib'] = sampler.device_peak
            measurements.append(m)
        # Images leave the worker only when deliberately requested for public/synthetic QA.
        if request.get('return_images', False):
            output['output_png'] = encode_png(result)
            frame = comparison_frame(face, aligned, aligned_ref, result)
            output['comparison_png'] = encode_png(frame)
        last = measurements[-1]
        output['metrics'] = {
            'status': 'ok', 'error_code': None,
            'inference_s': float(np.median([m['inference_s'] for m in measurements])),
            'pipeline_s': float(np.median([m['pipeline_s'] for m in measurements])),
            'alignment_s': float(np.median([m['alignment_s'] for m in measurements])),
            'decode_s': decode_s,
            'peak_allocated_mib': max(m['peak_allocated_mib'] for m in measurements),
            'peak_reserved_mib': max(m['peak_reserved_mib'] for m in measurements),
            'nvml_process_peak_mib': max((m['nvml_process_peak_mib'] for m in measurements if m['nvml_process_peak_mib'] is not None), default=None),
            'nvml_device_peak_mib': max((m['nvml_device_peak_mib'] for m in measurements if m['nvml_device_peak_mib'] is not None), default=None),
            'repeats': repetitions, 'measurements': measurements,
            'input_size': list(face.size), 'output_size': list(result.size),
            'alignment': 'ffhq_1024_crop' if isinstance(engine, HairEngine) else 'none',
            'backend': engine.backend, 'provenance': provenance,
        }
        if isinstance(engine, ColorEngine):
            output['metrics']['color_target_rgb'] = last['color_target_rgb']
            output['metrics']['hair_fraction'] = last['hair_fraction']
        return output
    finally:
        for image in (face, reference, result, aligned, aligned_ref, frame):
            if image is not None:
                wipe(image)
        engine.clear_private_state()
        # Request-owned mutable buffers are zeroed even on decode/OOM/model failure.
        for name in ('face', 'reference'):
            wipe(request.get(name))
            request.pop(name, None)
        gc.collect()
        torch.cuda.empty_cache()  # Releases unused blocks, not a secure erasure guarantee.


def read_exact(stream, n):
    value = bytearray(n)
    view = memoryview(value)
    pos = 0
    while pos < n:
        count = stream.readinto(view[pos:])
        if not count:
            wipe(value)
            raise EOFError('worker_pipe_closed')
        pos += count
    view.release()
    return value


def read_message(stream):
    header = read_exact(stream, 8)
    n = struct.unpack('!Q', header)[0]
    if n > 128 * 1024 * 1024:
        raise ValueError('ipc_size_limit')
    body = read_exact(stream, n)
    try:
        return pickle.loads(body)  # Trusted local parent only; not a network endpoint.
    finally:
        wipe(body)


def send_message(stream, message):
    body = bytearray(pickle.dumps(message, protocol=5))
    try:
        stream.write(struct.pack('!Q', len(body)))
        stream.write(body)
        stream.flush()
    finally:
        wipe(body)


def main():
    # stdout is reserved for framed IPC. All third-party prints go to stderr.
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    root = Path(os.environ['BEAUTY_ROOT'])
    repo = root / 'HairFastGAN'
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(2)
    service = sys.argv[1]
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        engine = HairEngine(root) if service == 'hairstyle' else ColorEngine(root)
        torch.cuda.synchronize()
        startup = {
            'model_load_s': time.perf_counter() - load_started,
            'init_peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'init_peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        }
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        provenance = json.loads((root / 'run_config.json').read_text())
        provenance.update(gpu=torch.cuda.get_device_name(), torch_runtime=torch.__version__,
                          cuda_runtime=torch.version.cuda, seed=3407, engine=service,
                          benchmark_profile='FP32_batch1_no_poisson')
        # Explicit public-sample warmup. Network is used only before user input.
        warmface = bytearray((root / 'public_demo' / '6.png').read_bytes())
        warmref = bytearray((root / 'public_demo' / '7.png').read_bytes())
        sys.addaudithook(audit_guard)
        warmup_started = time.perf_counter()
        for _ in range(2):
            with request_guard():
                evaluate(engine, {'face': warmface.copy(), 'reference': warmref.copy(), 'repeats': 1}, provenance)
        startup['warmup_s'] = time.perf_counter() - warmup_started
        provenance['startup'] = startup
        wipe(warmface)
        wipe(warmref)
        send_message(protocol_out, {'status': 'ready', 'provenance': provenance, 'backend': engine.backend})
    except Exception as exc:
        # Setup has no private images. Its traceback is useful for installation failures.
        import traceback
        traceback.print_exc(file=sys.stderr)
        send_message(protocol_out, {'status': 'init_failed', 'error_code': type(exc).__name__})
        return
    while True:
        request = response = None
        try:
            request = read_message(sys.stdin.buffer)
            if request.get('op') == 'stop':
                send_message(protocol_out, {'status': 'stopped'})
                break
            if request.get('op') != 'evaluate':
                raise ValueError('invalid_operation')
            with request_guard():
                response = evaluate(engine, request, provenance)
            send_message(protocol_out, response)
        except EOFError:
            break
        except Exception as exc:
            # No traceback or original filenames for private requests.
            code = 'cuda_out_of_memory' if isinstance(exc, torch.cuda.OutOfMemoryError) else type(exc).__name__
            known = {'exactly_one_face_required', 'insufficient_hair_pixels', 'insufficient_reference_hair',
                     'invalid_image_bytes', 'image_limits_exceeded', 'unsupported_image_format',
                     'side_effect_blocked_during_request', 'disk_write_blocked_during_request'}
            if str(exc) in known:
                code = str(exc)
            metrics = {'status': 'error', 'error_code': code, 'provenance': provenance}
            if request and request.get('debug_public'):
                import traceback
                metrics['public_diagnostic'] = traceback.format_exc()[-6000:]
            send_message(protocol_out, {'metrics': metrics})
        finally:
            if request is not None:
                wipe(request.get('face'))
                wipe(request.get('reference'))
                request.clear()
            if response is not None:
                wipe(response.get('output_png'))
                wipe(response.get('comparison_png'))
                response.clear()
            engine.clear_private_state()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
