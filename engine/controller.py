# Notebook control plane: stdlib only. Image arrays live in the isolated model worker.
import os, io, sys, json, time, gc, struct, pickle, subprocess, contextlib, zipfile, re, csv
from pathlib import Path, PurePosixPath
from collections import Counter
import select, threading
from collections import deque

PRIVATE_IMAGES = True       # Default: no image display, no return, no image export.
# Set False ONLY for public/synthetic benchmark images. Colab is not an audited private backend.
CLIENT = None
MAX_IPC = 128 * 1024 * 1024

def configure(root, code_root, python, env):
    global ROOT, CODE_ROOT, PYTHON, ENV
    ROOT, CODE_ROOT = Path(root), Path(code_root)
    PYTHON, ENV = str(python), dict(env)


def clear_buffer(buffer):
    if isinstance(buffer, bytearray):
        buffer[:] = b'\0' * len(buffer)
    elif isinstance(buffer, io.BytesIO):
        if buffer.closed:
            return
        view = buffer.getbuffer()
        try:
            view[:] = b'\0' * len(view)
        finally:
            view.release()
            buffer.close()


def recv_exact(stream, size, deadline=None):
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                clear_buffer(buffer)
                raise TimeoutError('model_response_timeout')
        n = stream.readinto(view[offset:])
        if not n:
            clear_buffer(buffer)
            raise RuntimeError('Model worker stopped; restart with start_engine().')
        offset += n
    view.release()
    return buffer


class ModelClient:
    def __init__(self, service):
        if service not in {'hairstyle', 'color'}:
            raise ValueError('Service must be hairstyle or color.')
        self.service = service
        self.initializing = True
        self.init_logs = deque(maxlen=60)
        self.process = subprocess.Popen([PYTHON, str(CODE_ROOT / 'worker.py'), service],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=ENV, bufsize=0)
        def drain():
            for raw in iter(self.process.stderr.readline, b''):
                if self.initializing:
                    line = raw.decode('utf-8', errors='replace')
                    self.init_logs.append(line)
                    print(line, end='', flush=True)
                # Request-time third-party output is discarded for private inputs.
        self.log_thread = threading.Thread(target=drain, daemon=True)
        self.log_thread.start()
        try:
            ready = self._receive(timeout=1800)
        except Exception as exc:
            self.close()
            raise RuntimeError('Model initialization failed:\n' + ''.join(self.init_logs)
                               + '\nTransport: ' + str(exc)) from exc
        if ready.get('status') != 'ready':
            self.close()
            raise RuntimeError('Model initialization failed:\n' + ''.join(self.init_logs))
        self.initializing = False
        self.ready = ready

    def _receive(self, timeout=180):
        deadline = time.monotonic() + timeout
        header = recv_exact(self.process.stdout, 8, deadline)
        size = struct.unpack('!Q', header)[0]
        if size > MAX_IPC:
            raise RuntimeError('IPC message too large.')
        body = recv_exact(self.process.stdout, size, deadline)
        try:
            return pickle.loads(body)
        finally:
            clear_buffer(body)

    def call(self, request):
        if self.process.poll() is not None:
            raise RuntimeError('Worker not running.')
        body = bytearray(pickle.dumps(request, protocol=5))
        try:
            # FileIO on pipes can short-write. Always write each frame completely.
            for part in (struct.pack('!Q', len(body)), body):
                view = memoryview(part)
                while view:
                    written = self.process.stdin.write(view)
                    if not written:
                        raise RuntimeError('Worker pipe closed.')
                    view = view[written:]
                view.release()
            self.process.stdin.flush()
            try:
                return self._receive()
            except Exception:
                self.close()
                raise
        finally:
            clear_buffer(body)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if hasattr(self, 'log_thread'):
            self.log_thread.join(timeout=1)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None:
                pipe.close()


def stop_engine():
    global CLIENT
    if CLIENT is not None:
        CLIENT.close()
        CLIENT = None
    gc.collect()


def start_engine(service='hairstyle'):
    """Keep ONE service resident on GPU. Warmup is public, not measured in scores."""
    global CLIENT
    stop_engine()
    CLIENT = ModelClient(service)
    print(json.dumps(CLIENT.ready, indent=2))


async_file_picker_js = r'''
new Promise(resolve => {
  const input = document.createElement('input');
  input.type = 'file'; input.accept = ACCEPT_TOKEN;
  input.style.display = 'block';
  const label = document.createElement('p'); label.textContent = LABEL_TOKEN;
  document.body.append(label, input);
  input.oncancel = () => {input.remove(); label.remove(); resolve(null);};
  input.onchange = async () => {
    const file = input.files[0];
    if (!file || file.size > LIMIT_TOKEN) {
      input.remove(); label.remove(); resolve({error:'invalid_file_size'}); return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const data = String(reader.result).split(',')[1];
      input.value = ''; input.remove(); label.remove(); resolve({data});
    };
    reader.onerror = () => {input.remove(); label.remove(); resolve({error:'read_failed'});};
    reader.readAsDataURL(file);
  };
})
'''


def pick_bytes(label='Choose image', zip_input=False):
    """FileReader -> memory. Does NOT use google.colab.files.upload (disk write)."""
    import base64
    from google.colab.output import eval_js
    script = async_file_picker_js.replace('ACCEPT_TOKEN', json.dumps('.zip' if zip_input else '.jpg,.jpeg,.png,.webp'))
    script = script.replace('LABEL_TOKEN', json.dumps(label)).replace('LIMIT_TOKEN', str(256 * 2**20 if zip_input else 25 * 2**20))
    response = eval_js(script, timeout_sec=300)
    try:
        if not response or response.get('error'):
            raise ValueError('No valid file selected.')
        return bytearray(base64.b64decode(response['data'], validate=True))
    finally:
        if response is not None:
            response.clear()
        gc.collect()


def show_comparison(png):
    if PRIVATE_IMAGES:
        raise PermissionError('Image display is disabled for private images: notebook outputs can persist.')
    from IPython.display import display, Image as DisplayImage
    display(DisplayImage(data=bytes(png)))


def evaluate(face_bytes, reference_bytes=None, *, show=False, retain_output=False,
             repeats=3, target_rgb=(116, 52, 40), strength=.85, reference_color=False, preview_fn=None):
    """Consumes/zeroes supplied bytearrays. Return only metrics by default.
    Pure inference excludes uploads/decode/render; pipeline_s includes alignment.
    """
    if CLIENT is None:
        raise RuntimeError('Run start_engine() first.')
    if PRIVATE_IMAGES and (show or retain_output):
        raise PermissionError('For private images, display/export/retention is blocked.')
    if not isinstance(face_bytes, bytearray) or (reference_bytes is not None and not isinstance(reference_bytes, bytearray)):
        raise TypeError('Use owned bytearray inputs so they can be overwritten.')
    response = None
    try:
        response = CLIENT.call({'op': 'evaluate', 'face': face_bytes, 'reference': reference_bytes,
                                'repeats': repeats, 'target_rgb': target_rgb, 'strength': strength,
                                'reference_color': reference_color, 'return_images': show or retain_output or preview_fn is not None,
                                'debug_public': not PRIVATE_IMAGES})
        if preview_fn is not None and response['metrics']['status'] == 'ok':
            preview_fn(response['comparison_png'])
        if show and response['metrics']['status'] == 'ok':
            show_comparison(response['comparison_png'])
        if retain_output:
            return response
        return response['metrics']
    finally:
        clear_buffer(face_bytes)
        clear_buffer(reference_bytes)
        if response is not None and not retain_output:
            clear_buffer(response.get('output_png'))
            clear_buffer(response.get('comparison_png'))
        gc.collect()


def single_demo(*, show=False, reference_color=False):
    # No upload data assigned to notebook globals or returned as last-cell output.
    face = reference = None
    try:
        face = pick_bytes('Choose target portrait')
        if CLIENT is None:
            raise RuntimeError('Run start_engine first.')
        if CLIENT.service == 'hairstyle' or reference_color:
            reference = pick_bytes('Choose hair reference (one face)')
        metrics = evaluate(face, reference, show=show, reference_color=reference_color)
        print(json.dumps(metrics, indent=2))
    finally:
        clear_buffer(face)
        clear_buffer(reference)


ID_PATTERN = re.compile(r'[ABC](0[1-9]|10)')


def validate_manifest(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 30:
        raise ValueError('Manifest must contain 1..30 cases.')
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not ID_PATTERN.fullmatch(str(case.get('id', ''))):
            raise ValueError('Use anonymous IDs A01..A10, B01..B10, C01..C10.')
        if case['id'] in ids or case.get('group') != case['id'][0]:
            raise ValueError('Duplicate ID or inconsistent group.')
        ids.add(case['id'])
        for key in ('target', 'reference'):
            path = case.get(key)
            if key == 'reference' and path is None:
                continue
            if not isinstance(path, str) or not path or '\\' in path:
                raise ValueError('Invalid manifest image path.')
            pure = PurePosixPath(path)
            if pure.is_absolute() or '..' in pure.parts or pure.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:
                raise ValueError('Invalid manifest image path.')
    return cases


@contextlib.contextmanager
def zip_cases(zip_bytes):
    """Never extract. Drop compressed image members immediately after last use."""
    stream = io.BytesIO(zip_bytes)
    clear_buffer(zip_bytes)
    try:
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            names = [x.filename for x in infos]
            if len(names) > 256 or len(names) != len(set(names)):
                raise ValueError('Invalid ZIP entry count / duplicate entries.')
            if sum(x.file_size for x in infos) > 512 * 2**20:
                raise ValueError('ZIP expansion limit exceeded.')
            if any(x.flag_bits & 1 or x.file_size > 25 * 2**20 or
                   x.file_size / max(1, x.compress_size) > 200 for x in infos):
                raise ValueError('ZIP entry limits exceeded.')
            if 'manifest.json' not in names:
                raise ValueError('ZIP must include manifest.json at its root.')
            info = archive.getinfo('manifest.json')
            if info.file_size > 128 * 1024:
                raise ValueError('Manifest too large.')
            cases = validate_manifest(json.loads(archive.read('manifest.json')))
            needed = [case[key] for case in cases for key in ('target', 'reference') if case.get(key)]
            if any(name not in names for name in needed):
                raise ValueError('Manifest refers to a missing image.')
            remaining = Counter(needed)
            def loader(name):
                return bytearray(archive.read(name))
            def release(name):
                remaining[name] -= 1
                if remaining[name] == 0:
                    entry = archive.getinfo(name)
                    view = stream.getbuffer()
                    start = entry.header_offset
                    header = struct.unpack_from('<4s5H3I2H', view, start)
                    body_start = start + 30 + header[-2] + header[-1]
                    view[body_start:body_start + entry.compress_size] = b'\0' * entry.compress_size
                    view.release()
            yield cases, loader, release
    finally:
        clear_buffer(stream)
        gc.collect()


def record_scores(case, metrics):
    """Human QA after public or transient private preview; never inferred from latency."""
    print('Rate', case['id'], ': 1=unusable, 2=major errors, 3=visible issues, 4=acceptable, 5=excellent')
    while True:
        raw = input('Identity, Hairline, Reference (e.g. 4 4 5): ').split()
        try:
            values = [int(x) for x in raw]
            if len(values) == 3 and all(1 <= x <= 5 for x in values):
                return dict(zip(('identity', 'hairline', 'reference'), values))
        except ValueError:
            pass
        print('Enter three integer scores from 1 to 5.')


def run_batch(cases, loader, *, release=None, show=False, score_fn=None,
              save_dir=None, allow_public_export=False, repeats=3, reference_color=False, preview_fn=None):
    """Sequential batch, no model reload/cache. Sink runs before output-buffer cleanup.
    Private previews use the transient callback; exports stay blocked. Failures count.
    """
    validate_manifest(cases)
    if PRIVATE_IMAGES and (show or save_dir is not None):
        raise PermissionError('Private-image display/export is disabled.')
    if save_dir is not None and not allow_public_export:
        raise PermissionError('Set allow_public_export=True only for public/synthetic data.')
    if CLIENT is None:
        raise RuntimeError('Start an engine before running a batch.')
    records = []
    for case in cases:
        face = reference = response = None
        row = {'id': case['id'], 'group': case['group'], 'engine': CLIENT.service,
               'identity': None, 'hairline': None, 'reference': None}
        try:
            face = loader(case['target'])
            reference = loader(case['reference']) if case.get('reference') else None
            response = evaluate(face, reference, show=show, retain_output=save_dir is not None,
                                repeats=repeats, reference_color=reference_color, preview_fn=preview_fn)
            metrics = response['metrics'] if save_dir is not None else response
            row.update(metrics)
            if metrics['status'] == 'ok':
                if save_dir is not None:
                    dest = Path(save_dir)
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / (case['id'] + '_result.png')).write_bytes(response['output_png'])
                    (dest / (case['id'] + '_comparison.png')).write_bytes(response['comparison_png'])
                    print('Public comparison ready:', dest / (case['id'] + '_comparison.png'))
                if score_fn is not None:
                    ratings = score_fn(case, metrics)
                    if ratings:
                        for key in ('identity', 'hairline', 'reference'):
                            if isinstance(ratings.get(key), bool) or not isinstance(ratings.get(key), int) or not 1 <= ratings[key] <= 5:
                                raise ValueError('Invalid score.')
                        row.update(ratings)
        except Exception as exc:
            row.update(status='error', error_code=type(exc).__name__)
        finally:
            clear_buffer(face)
            clear_buffer(reference)
            if isinstance(response, dict):
                clear_buffer(response.get('output_png'))
                clear_buffer(response.get('comparison_png'))
            if release:
                for key in ('target', 'reference'):
                    if case.get(key):
                        release(case[key])
        records.append(row)
        print(case['id'], row['status'], 'inference_s=', row.get('inference_s'),
              'peak_MiB=', row.get('peak_allocated_mib'), 'error=', row.get('error_code'))
    return records


def batch_zip_demo(*, show=False, score_fn=None, save_dir=None, allow_public_export=False, preview_fn=None):
    blob = pick_bytes('Choose ZIP with manifest.json and images', zip_input=True)
    try:
        with zip_cases(blob) as (cases, loader, release):
            return run_batch(cases, loader, release=release, show=show, score_fn=score_fn,
                             save_dir=save_dir, allow_public_export=allow_public_export, preview_fn=preview_fn)
    finally:
        clear_buffer(blob)


def batch_folder(folder, *, show=False, score_fn=None, save_dir=None, allow_public_export=False, preview_fn=None):
    """Read a pre-existing folder without modifying input files. PUBLIC DATA ONLY.
    Strict private runs use memory upload instead of creating folders on Colab.
    """
    if PRIVATE_IMAGES:
        raise PermissionError('Folder mode is restricted to pre-existing public/synthetic datasets.')
    root = Path(folder).resolve()
    cases = validate_manifest(json.loads((root / 'manifest.json').read_text()))
    def loader(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 25 * 2**20:
            raise ValueError('Invalid image path.')
        return bytearray(path.read_bytes())
    return run_batch(cases, loader, show=show, score_fn=score_fn, save_dir=save_dir,
                     allow_public_export=allow_public_export, preview_fn=preview_fn)


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    k = (len(ordered) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo) if hi != lo else ordered[lo]


def quality_gate(records, *, threshold=4, seconds=8., latency_key='inference_s'):
    """Provisional preregistered rubric: ALL 3 quality scores >=4.
    Gate: full 30 (10/group), >=21 quality passes AND >=21 joint quality/latency
    passes. P95 is reported separately, not added to the user's gate. Failures count in /30.
    Missing ratings => UNASSESSED. Different engines may not be pooled.
    """
    expected_ids = {f'{g}{i:02d}' for g in 'ABC' for i in range(1, 11)}
    actual = [r.get('id') for r in records]
    engines = {r.get('engine') for r in records}
    full = len(actual) == 30 and set(actual) == expected_ids and len(set(actual)) == 30
    full = full and all(r.get('group') == r['id'][0] for r in records) and len(engines) == 1
    if not 1 <= threshold <= 5 or seconds <= 0:
        raise ValueError('Invalid gate threshold.')
    quality = joint = 0
    missing_latency = 0
    missing = 0
    groups = {g: {'total': 0, 'quality_pass': 0, 'joint_pass': 0, 'errors': 0} for g in 'ABC'}
    times = []
    for row in records:
        group = groups.get(row.get('group'))
        if group is None:
            continue
        group['total'] += 1
        if row.get('status') != 'ok':
            group['errors'] += 1
            continue
        latency = row.get(latency_key)
        if isinstance(latency, (int, float)) and math.isfinite(latency) and latency >= 0:
            times.append(latency)
        else:
            latency = None
            missing_latency += 1
        values = [row.get(k) for k in ('identity', 'hairline', 'reference')]
        valid = all(isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= 5 for x in values)
        if not valid:
            missing += 1
            continue
        good = min(values) >= threshold and not row.get('identity_hard_fail', False)
        fast = latency is not None and latency < seconds
        quality += int(good)
        joint += int(good and fast)
        group['quality_pass'] += int(good)
        group['joint_pass'] += int(good and fast)
    p95 = percentile(times, 95)
    if not full:
        status = 'INCOMPLETE'
    elif missing or missing_latency:
        status = 'UNASSESSED'
    elif quality >= 21 and joint >= 21:
        status = 'PASS'
    else:
        status = 'FAIL'
    return {'gate': status, 'full_protocol': full, 'quality_passes': quality,
            'joint_passes': joint, 'denominator': 30, 'quality_pass_rate': quality / 30,
            'joint_pass_rate': joint / 30, 'unrated_successes': missing,
            'missing_latency': missing_latency,
            'latency_metric': latency_key, 'median_s': percentile(times, 50), 'p95_s': p95,
            'seconds_limit_exclusive': seconds, 'groups': groups,
            'note': 'Scores >=4 are a proposed operational rubric; agree before collecting results.'}


# Non-image reports only: no source paths, original names, photos or latent embeddings.
import math

def export_metrics(records, path='/content/beauty_phase0/metrics.csv', seconds=8.):
    keys = ['id','group','engine','status','error_code','identity','hairline','reference',
            'inference_s','alignment_s','pipeline_s','decode_s','peak_allocated_mib',
            'peak_reserved_mib','nvml_process_peak_mib','nvml_device_peak_mib','repeats']
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    metadata_path = str(Path(path).with_suffix('.json'))
    with open(metadata_path, 'w') as stream:
        json.dump({'gate': quality_gate(records, seconds=seconds), 'records': records}, stream, indent=2)
    print('Saved anonymous metrics CSV and JSON:', path, metadata_path)
