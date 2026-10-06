import json, time
from core import np, cv2, Image, wipe, lab_recolor

class ColorEngine:
    def __init__(self, root, side=512):
        import torch
        from transformers import SegformerForSemanticSegmentation
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            str(root / 'segformer'), local_files_only=True, use_safetensors=True).eval().cuda()
        self.side = side
        config = json.loads((root / 'segformer/preprocessor_config.json').read_text())
        self.mean = torch.tensor(config['image_mean'], device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor(config['image_std'], device='cuda').view(1, 3, 1, 1)
        labels = self.model.config.id2label
        hits = [int(key) for key, value in labels.items() if value.lower() == 'hair']
        if len(hits) != 1:
            raise RuntimeError('hair_label_missing')
        self.hair_id = hits[0]
        self.backend = {'segformer': 'torch_fp32', 'input_size': side, 'hair_label': self.hair_id}

    def mask(self, rgb):
        import torch
        import torch.nn.functional as F
        small = cv2.resize(rgb, (self.side, self.side), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
        logits = probs = hair = hard = confidence = None
        array = hard_array = None
        try:
            logits = self.model(pixel_values=(x - self.mean) / self.std).logits
            probs = F.interpolate(logits, size=rgb.shape[:2], mode='bilinear', align_corners=False).softmax(1)
            hair = probs[:, self.hair_id]
            # Never dilate hair onto face. Soften only inside the hard hair support.
            hard = probs.argmax(1) == self.hair_id
            confidence = ((hair - .35) / .5).clamp(0, 1) * hard
            array = confidence[0].cpu().numpy().copy()
            hard_array = hard[0].cpu().numpy()
            soft = cv2.GaussianBlur(array, (3, 3), 0.7)
            soft *= hard_array
            soft[soft < .02] = 0
            wipe(array)
            return soft
        finally:
            wipe(small)
            for a in (array, hard_array):
                if a is not None:
                    wipe(a)
            for t in (x, logits, probs, hair, hard, confidence):
                if t is not None:
                    t.zero_()

    def clear_private_state(self):
        pass

    def run(self, face, reference=None, target_rgb=(116, 52, 40), strength=.85, reference_color=False):
        rgb = np.array(face)
        mask = ref_rgb = ref_mask = out = None
        t0 = time.perf_counter()
        try:
            mask = self.mask(rgb)
            if reference_color:
                if reference is None:
                    raise ValueError('color_reference_required')
                ref_rgb = np.array(reference)
                ref_mask = self.mask(ref_rgb)
                selected = ref_mask >= .75
                if selected.sum() < 32:
                    raise ValueError('insufficient_reference_hair')
                # Approximate median RGB swatch; lighting affects this measurement.
                target_rgb = np.median(ref_rgb[selected], axis=0).tolist()
            out = lab_recolor(rgb, mask, target_rgb, strength)
            import torch
            torch.cuda.synchronize()
            inference_s = time.perf_counter() - t0
            return Image.fromarray(out.copy()), face.copy(), (reference or face).copy(), {
                'alignment_s': 0., 'inference_s': inference_s,
                'color_target_rgb': target_rgb, 'hair_fraction': float((mask > 0).mean()),
                'reference_color': reference_color,
            }
        finally:
            for x in (rgb, mask, ref_rgb, ref_mask, out):
                if x is not None:
                    wipe(x)
