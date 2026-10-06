"""Notebook orchestration. Image requests remain in controller/worker memory."""
import contextlib, hashlib, importlib.util, json, os, pathlib, subprocess, sys, time
from collections import deque

ROOT = pathlib.Path(os.environ.get('BEAUTY_ROOT', '/content/beauty_phase0')).resolve()
CODE_ROOT = pathlib.Path(__file__).resolve().parent
TEST_ROOT = pathlib.Path(os.environ.get('BEAUTY_TEST_ROOT', str(CODE_ROOT.parent / 'tests')))
REPO = ROOT / 'HairFastGAN'
VENV = ROOT / 'venv'
PYTHON = str(VENV / 'bin/python')
CODE_REV = '49e98019c9243d1991c19d45764fc3340a498de2'
WEIGHTS_REV = '47855f8ea72b02294649385211a1ae1fd7dcb58e'
SEG_MODEL = 'isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing'
SEG_REV = 'dd02ee5b73017377ba6f12a58c3e300cbbfd58c8'
OP_BACKEND = os.environ.get('BEAUTY_OP_BACKEND', 'auto')
ENV = dict(os.environ, BEAUTY_ROOT=str(ROOT), BEAUTY_OP_BACKEND=OP_BACKEND,
           BEAUTY_TEST_DEMO=str(ROOT/'public_demo'),
           WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           PYTHONDONTWRITEBYTECODE='1', MAX_JOBS='2',
           TORCH_CUDA_ARCH_LIST='7.5;8.9')

def run(*command, env=None, timeout=1800):
    """Stream public setup/test output; include its actual error in the exception."""
    command = list(map(str, command))
    tail = deque(maxlen=60)
    child_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                     CMAKE_BUILD_PARALLEL_LEVEL='2', MAX_JOBS='2')
    child_env['PATH'] = str(VENV / 'bin') + os.pathsep + child_env.get('PATH', '')
    if env:
        child_env.update(env)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1, env=child_env) as child:
        # A watchdog bounds stalled setup even when no new output arrives.
        import threading
        expired = threading.Event()
        def terminate():
            expired.set()
            child.kill()
        timer = threading.Timer(timeout, terminate)
        timer.daemon = True
        timer.start()
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                tail.append(line)
            code = child.wait()
        finally:
            timer.cancel()
    if expired.is_set():
        raise TimeoutError('Setup/test timeout:\n' + ''.join(tail))
    if code:
        raise RuntimeError('Command failed (exit %s):\n%s' % (code, ''.join(tail)))

def write_json(path, value):
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)

@contextlib.contextmanager
def stage(number, title):
    """Persist anonymous stage status; never photos, masks, input names or latents."""
    path = CODE_ROOT / 'run_status.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    item = {'status':'RUNNING', 'title':title, 'started_at':time.time()}
    state[str(number)] = item
    write_json(path, state)
    print('\n[%s] %s — شروع' % (number, title), flush=True)
    began = time.perf_counter()
    try:
        yield
    except Exception as exc:
        item.update(status='FAILED', seconds=time.perf_counter()-began,
                    error_type=type(exc).__name__)
        write_json(path, state)
        print('[%s] متوقف شد؛ علت خطا در خروجی همین سلول است.' % number, flush=True)
        raise
    else:
        item.update(status='PASSED', seconds=time.perf_counter()-began)
        write_json(path, state)
        print('[%s] کامل شد (%.1f ثانیه)' % (number, item['seconds']), flush=True)

def preflight():
    import shutil
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('Runtime → Change runtime type → T4 GPU را انتخاب کن.')
    run('nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv')
    if shutil.disk_usage(ROOT).free < 18 * 2**30 and not (ROOT/'weight_manifest.json').exists():
        raise RuntimeError('برای نصب اول حداقل ۱۸ GiB فضای آزاد لازم است.')
    print('Python مدل مستقل از Python فعلی Colab ساخته می‌شود.', flush=True)

def _script(name):
    run(sys.executable, str(CODE_ROOT/name))

def install():
    _script('install.py')

def assets():
    _script('assets.py')

def verify():
    if not pathlib.Path(PYTHON).is_file():
        raise RuntimeError('سلول ۴ نصب را اجرا کن؛ محیط مدل موجود نیست.')
    probe = '''
import torch, torchvision, numpy, cv2, dlib, clip, transformers, pkg_resources
assert torch.cuda.is_available(), "CUDA unavailable"
x=torch.ones((16,16),device="cuda"); y=x@x; torch.cuda.synchronize()
assert y[0,0].item()==16
print("CUDA arithmetic passed:",torch.cuda.get_device_name())
print("Torch:",torch.__version__,"NumPy:",numpy.__version__,"OpenCV:",cv2.__version__)
'''
    run(PYTHON, '-c', probe)
    run(PYTHON, str(TEST_ROOT/'test_core.py'), env=ENV)
    run(PYTHON, str(TEST_ROOT/'test_runtime.py'), env=ENV)
    run(PYTHON, str(TEST_ROOT/'test_operators.py'),
        env=dict(ENV, BEAUTY_TEST_REPO=str(REPO)))

def api():
    name = '_beauty_benchmark_api'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, CODE_ROOT/'controller.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    module.configure(ROOT, CODE_ROOT, PYTHON, ENV)
    return module

def demo(service, show=True):
    if service not in {'hairstyle','color'}:
        raise ValueError('Unknown service')
    module = api()
    face = reference = response = None
    previous = module.PRIVATE_IMAGES
    try:
        module.start_engine(service)
        module.PRIVATE_IMAGES = False  # Official PUBLIC images only.
        face = bytearray((ROOT/'public_demo/6.png').read_bytes())
        reference = bytearray((ROOT/'public_demo/7.png').read_bytes())
        response = module.evaluate(face, reference, show=show, retain_output=not show, repeats=3)
        metrics = response if show else response['metrics']
        if metrics['status'] != 'ok':
            print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
            raise RuntimeError(metrics.get('public_diagnostic') or metrics['error_code'])
        write_json(CODE_ROOT/(service+'_demo_metrics.json'), metrics)
        if not show:
            destination = ROOT/'public_demo_outputs'
            destination.mkdir(parents=True, exist_ok=True)
            (destination/(service+'_result.png')).write_bytes(response['output_png'])
            (destination/(service+'_comparison.png')).write_bytes(response['comparison_png'])
            print('Public demo comparison:', destination/(service+'_comparison.png'))
        print('مدل:',service, '| زمان خالص:',round(metrics['inference_s'],3),'s',
              '| زمان پایپ‌لاین:',round(metrics['pipeline_s'],3),'s',
              '| Peak VRAM:',round(metrics['peak_allocated_mib'],1),'MiB',flush=True)
        return metrics
    finally:
        module.clear_buffer(face)
        module.clear_buffer(reference)
        if response is not None:
            module.clear_buffer(response.get('output_png'))
            module.clear_buffer(response.get('comparison_png'))
        module.PRIVATE_IMAGES = previous
        module.stop_engine()

def summary():
    from IPython.display import display, Markdown
    rows = []
    incomplete = []
    state_path=CODE_ROOT/'run_status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    for stage_number, service, limit in [(17,'hairstyle',8.0),(18,'color',0.5)]:
        path = CODE_ROOT/(service+'_demo_metrics.json')
        if not path.is_file() or state.get(str(stage_number),{}).get('status')!='PASSED':
            rows.append('|%s|اجرا نشده|—|—|—|—|' % service)
            incomplete.append(str(stage_number))
            continue
        data = json.loads(path.read_text())
        rows.append('|%s|%s|%.3f|%.3f|%.1f|%s|' % (
            service,data['status'],data['inference_s'],data['pipeline_s'],data['peak_allocated_mib'],
            'زیر هدف زمان دمو' if data['inference_s'] < limit else 'بالاتر از هدف زمان دمو'))
    display(Markdown('| موتور | وضعیت | زمان خالص (s) | پایپ‌لاین (s) | Peak VRAM (MiB) | زمان دمو |\n'
                     '|---|---|---:|---:|---:|---|\n'+'\n'.join(rows)))
    print('این جدول نتیجهٔ دموی عمومی است؛ گیت ۳۰ عکس هنوز ارزیابی نشده است.',flush=True)
    if incomplete:
        raise RuntimeError('دموی کامل تأیید نشد؛ سلول‌های ' + ' و '.join(incomplete)
                           + ' هنوز در همین اجرا موفق نشده‌اند.')

def sample(service='hairstyle', public=False):
    module=api(); previous=module.PRIVATE_IMAGES
    try:
        module.PRIVATE_IMAGES=not public
        module.start_engine(service)
        module.single_demo(show=public)
    finally:
        module.PRIVATE_IMAGES=previous
        module.stop_engine()

def batch(service='hairstyle', public=False, folder=None, save_outputs=False, review=True):
    module=api(); previous=module.PRIVATE_IMAGES
    try:
        module.PRIVATE_IMAGES=not public
        module.start_engine(service)
        preview_fn = None
        if review and not public:
            from photo_preview import show_transient
            preview_fn = show_transient
        options=dict(show=public,score_fn=module.record_scores if public or review else None,
                     preview_fn=preview_fn,
                     save_dir=str(ROOT/'public_benchmark_outputs') if save_outputs else None,
                     allow_public_export=public and save_outputs)
        records=module.batch_folder(folder,**options) if folder else module.batch_zip_demo(**options)
        gate=module.quality_gate(records,seconds=8.0 if service=='hairstyle' else .5)
        print(json.dumps(gate,ensure_ascii=False,indent=2),flush=True)
        module.export_metrics(records,path=str(CODE_ROOT/(service+'_benchmark.csv')),
                              seconds=8.0 if service=='hairstyle' else .5)
        return gate
    finally:
        module.PRIVATE_IMAGES=previous
        module.stop_engine()

def stop():
    api().stop_engine()
