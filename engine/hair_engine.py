import sys, time
from core import np, wipe

class HairEngine:
    def __init__(self, root):
        import torch, dlib, clip
        import functools
        # All downloaded public weights remain on disk. Client images never do.
        original_load = clip.load
        clip.load = functools.partial(original_load, download_root=str(root / 'clip'))
        from hair_swap import HairFast, get_parser
        from utils.shape_predictor import align_face
        args = get_parser().parse_args([])
        args.save_all = False
        args.batch_size = 1  # Sequential embeddings lower the T4 memory requirement.
        self.model = HairFast(args)
        # CLIP's official loader chooses FP16 on CUDA by default; normalize this
        # benchmark profile explicitly rather than mislabeling mixed precision.
        for module in (self.model.net, self.model.embed, self.model.align, self.model.blend):
            module.float().eval()
        self.predictor = dlib.shape_predictor('pretrained_models/ShapeAdaptor/shape_predictor_68_face_landmarks.dat')
        self.detector = dlib.get_frontal_face_detector()
        self.align_face = align_face
        self.backend = {}
        for name, module in list(sys.modules.items()):
            if name.endswith('.op.fused_act') or name.endswith('.op.upfirdn2d'):
                variable = 'fused' if name.endswith('fused_act') else 'upfirdn2d_op'
                self.backend[name] = 'cuda_fused' if getattr(module, variable, None) is not None else 'torch_reference'
        # Disable any optional SEAN latent-code exports.
        for sub in self.model.align.sean_model.modules():
            if hasattr(sub, 'save_npy'):
                sub.save_npy = False

    def clear_private_state(self):
        # Upstream Embedding retains its last dataset/dataloader even after return.
        embed = self.model.embed
        data = getattr(embed, 'dataset', None)
        if data is not None:
            import torch
            for image in getattr(data, 'images', []):
                if isinstance(image, torch.Tensor):
                    image.zero_()
            data.images = []
            data.names = []
        for attr in ('dataset', 'dataloader'):
            if hasattr(embed, attr):
                delattr(embed, attr)

    def run(self, face, reference):
        import torch
        import torchvision.transforms.functional as TF
        aligned = []
        returned = None
        try:
            # Reject ambiguous multi-face inputs rather than silently choosing a face.
            for image in (face, reference):
                array = np.array(image)
                try:
                    if len(self.detector(array, 1)) != 1:
                        raise ValueError('exactly_one_face_required')
                finally:
                    wipe(array)
            t0 = time.perf_counter()
            aligned = self.align_face([face, reference], predictor=self.predictor, return_tensors=True)
            torch.cuda.synchronize()
            alignment_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            returned = self.model.swap(aligned[0], aligned[1], aligned[1], align=False,
                                       benchmark=False, seed=3407)
            torch.cuda.synchronize()
            inference_s = time.perf_counter() - t1
            result = TF.to_pil_image(returned.detach().cpu())
            aligned_face = TF.to_pil_image(aligned[0])
            aligned_ref = TF.to_pil_image(aligned[1])
            return result, aligned_face, aligned_ref, {'alignment_s': alignment_s, 'inference_s': inference_s}
        finally:
            if returned is not None:
                returned.zero_()
            for tensor in aligned:
                tensor.zero_()
            self.clear_private_state()
