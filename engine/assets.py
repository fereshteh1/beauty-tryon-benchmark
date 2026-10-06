import ast, json, pathlib, sys, os, shutil
from runtime import ROOT, REPO, VENV, PYTHON, CODE_ROOT, CODE_REV, WEIGHTS_REV, SEG_MODEL, SEG_REV, OP_BACKEND, run
if not (REPO/'.git').is_dir():
    run('git','clone','https://github.com/AIRI-Institute/HairFastGAN.git',REPO)
run('git','-C',REPO,'fetch','origin',CODE_REV)
run('git','-C',REPO,'checkout','--force',CODE_REV)
patches = []
root_upfirdn = (REPO / 'models/stylegan2/op/upfirdn2d.py').read_text()
native_upfirdn = root_upfirdn[root_upfirdn.index('\ndef upfirdn2d_native('):]
for path in REPO.rglob('op/*.py'):
    if path.name not in {'fused_act.py', 'upfirdn2d.py'}:
        continue
    source = path.read_text()
    if path.name == 'upfirdn2d.py' and 'from torch.nn import functional as F' not in source:
        # The FSE copy has a native implementation but omitted its functional import.
        source = 'from torch.nn import functional as F\n' + source
    tree = ast.parse(source)
    node = next((n for n in tree.body if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == 'load'), None)
    if node is None:
        raise RuntimeError('Upstream operator structure changed: ' + str(path.relative_to(REPO)))
    variable = node.targets[0].id
    lines = source.splitlines(keepends=True)
    original = ''.join(lines[node.lineno - 1:node.end_lineno])
    guarded = ('try:\n'
               "    if os.environ.get('BEAUTY_OP_BACKEND') == 'torch_reference':\n"
               "        raise RuntimeError('reference_backend_requested')\n"
               + ''.join('    ' + line for line in original.splitlines(keepends=True))
               + '\nexcept Exception:\n    ' + variable + ' = None\n')
    source = ''.join(lines[:node.lineno - 1]) + guarded + ''.join(lines[node.end_lineno:])
    if path.name == 'fused_act.py':
        start = source.index('\ndef fused_leaky_relu(')
        source = source[:start] + '''
def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    if fused is not None and input.device.type == 'cuda':
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    rest = [1] * (input.ndim - bias.ndim - 1)
    return torch.nn.functional.leaky_relu(input + bias.view(1, bias.shape[0], *rest), negative_slope) * scale
'''
    else:
        start = source.index('\ndef upfirdn2d(')
        end = source.index('\ndef upfirdn2d_native(', start)
        source = source[:start] + '''
def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if upfirdn2d_op is not None and input.device.type == 'cuda':
        return UpFirDn2d.apply(input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1]))
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

''' + native_upfirdn
    ast.parse(source)
    path.write_text(source)
    patches.append(str(path.relative_to(REPO)))
# Fix removed Pillow alias in official alignment code; no resize-geometry change.
for path in REPO.rglob('*.py'):
    source = path.read_text()
    if 'PIL.Image.ANTIALIAS' in source:
        path.write_text(source.replace('PIL.Image.ANTIALIAS', 'PIL.Image.Resampling.LANCZOS'))


WEIGHTS = [
    'ArcFace/backbone_ir50.pth', 'BiSeNet/face_parsing_79999_iter.pth',
    'Blending/checkpoint.pth', 'FeatureStyleEncoder/143_enc.pth',
    'FeatureStyleEncoder/79999_iter.pth', 'FeatureStyleEncoder/backbone.pth',
    'FeatureStyleEncoder/psp_ffhq_encode.pt', 'PostProcess/latent_avg.pt',
    'PostProcess/pp_model.pth', 'Rotate/rotate_best.pth', 'ShapeAdaptor/mask_generator.pth',
    'ShapeAdaptor/shape_predictor_68_face_landmarks.dat', 'StyleGAN/ffhq.pt',
    'StyleGAN/ffhq_PCA.npz', 'encoder4editing/e4e_ffhq_encode.pt',
    'sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth',
]
download_code = '''
from huggingface_hub import snapshot_download
import pathlib, json, hashlib, clip
repo, revision, seg_model, seg_revision, root, files = json.loads(__import__('sys').argv[1])
snapshot_download(repo_id='AIRI-Institute/HairFastGAN', revision=revision,
                  allow_patterns=['pretrained_models/' + f for f in files], local_dir=repo,
                  max_workers=2)
snapshot_download(repo_id='AIRI-Institute/HairFastGAN', revision=revision,
                  allow_patterns=['input/6.png','input/7.png'], local_dir=repo, max_workers=2)
demo = pathlib.Path(root) / 'public_demo'
demo.mkdir(exist_ok=True)
for name in ('6.png','7.png'):
    __import__('shutil').copyfile(pathlib.Path(repo)/'input'/name, demo/name)
snapshot_download(repo_id=seg_model, revision=seg_revision,
                  allow_patterns=['config.json','preprocessor_config.json','model.safetensors'],
                  local_dir=root + '/segformer', max_workers=2)
# Force public CLIP weight download before the strict offline request guard is armed.
clip.load('ViT-B/32', device='cpu', download_root=root + '/clip')
manifest = {}
for file in files:
    p = pathlib.Path(repo) / 'pretrained_models' / file
    if not p.is_file() or p.stat().st_size < 1024:
        raise RuntimeError('Missing/invalid model asset: ' + file)
    with p.open('rb') as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    manifest[file] = {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}
(pathlib.Path(root) / 'weight_manifest.json').write_text(json.dumps(manifest, indent=2))
'''
run(PYTHON, '-c', download_code, json.dumps([str(REPO), WEIGHTS_REV, SEG_MODEL, SEG_REV, str(ROOT), WEIGHTS]))
(ROOT / 'run_config.json').write_text(json.dumps({
    'code_revision': CODE_REV, 'weights_revision': WEIGHTS_REV,
    'segformer_model': SEG_MODEL, 'segformer_revision': SEG_REV,
    'operator_backend_requested': OP_BACKEND, 'patches': patches,
    'python': '3.10', 'torch': '2.2.2+cu121', 'precision': 'float32',
}, indent=2))

config=json.loads((ROOT/'run_config.json').read_text())
config['notebook_version']='beauty_tryon_benchmark_0.1.0'
config['resolved_requirements_sha256']=__import__('hashlib').sha256((CODE_ROOT/'requirements.resolved.txt').read_bytes()).hexdigest()
(ROOT/'run_config.json').write_text(json.dumps(config,indent=2))
cfg=json.loads((ROOT/'segformer/config.json').read_text())
assert [int(k) for k,v in cfg['id2label'].items() if v.lower()=='hair']==[2]
assert len(patches)==6, 'All six StyleGAN operator copies must be patched'
print('16 HairFastGAN assets, segmentation configuration and six operators prepared.',flush=True)
