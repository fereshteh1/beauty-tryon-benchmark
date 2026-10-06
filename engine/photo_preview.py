"""Colab memory upload and temporary preview; no image files or display_data.
Python/browser/Colab copies cannot be securely erased or independently audited.
"""
import base64
import gc
import json

def pick_image(label):
    from google.colab.output import eval_js
    script = r'''
    new Promise(resolve => {
      const box = document.createElement('section');
      box.dir = 'rtl';
      const title = document.createElement('p'); title.textContent = LABEL;
      const input = document.createElement('input');
      input.type = 'file'; input.accept = '.jpg,.jpeg,.png,.webp';
      const cancel = document.createElement('button'); cancel.textContent = 'لغو';
      box.append(title, input, cancel); document.body.append(box);
      let done = false; let reader = null;
      const finish = value => {
        if (done) return; done = true; clearTimeout(timer);
        if (reader && reader.readyState === 1) reader.abort();
        if (reader) {reader.onload = null; reader.onerror = null; reader = null;}
        input.value = ''; box.remove(); resolve(value);
      };
      const timer = setTimeout(() => finish({error:'timeout'}), 290000);
      cancel.onclick = () => finish({error:'cancelled'});
      input.oncancel = () => finish({error:'cancelled'});
      input.onchange = () => {
        const file = input.files[0];
        if (!file || file.size === 0 || file.size > 25 * 1024 * 1024) {
          finish({error:'size'}); return;
        }
        reader = new FileReader();
        reader.onerror = () => finish({error:'read_failed'});
        reader.onload = () => {
          const data = String(reader.result).split(',')[1];
          finish({data});
        };
        reader.readAsDataURL(file);
      };
    })
    '''.replace('LABEL', json.dumps(label, ensure_ascii=True))
    reply = None
    try:
        reply = eval_js(script, timeout_sec=300)
        if not reply or reply.get('error'):
            messages = {
                'size': 'عکس باید JPEG، PNG یا WEBP و حداکثر ۲۵ مگابایت باشد.',
                'read_failed': 'مرورگر نتوانست عکس را بخواند؛ دوباره همین سلول را اجرا کن.',
                'timeout': 'زمان انتخاب عکس تمام شد؛ دوباره همین سلول را اجرا کن.',
                'cancelled': 'انتخاب عکس لغو شد.',
            }
            raise ValueError(messages.get((reply or {}).get('error'), 'عکسی انتخاب نشد.'))
        return bytearray(base64.b64decode(reply['data'], validate=True))
    finally:
        if reply is not None:
            reply.clear()
        gc.collect()

def show_transient(png):
    from google.colab.output import eval_js
    encoded = script = None
    try:
        encoded = base64.b64encode(png).decode('ascii')
        script = r'''
        (() => {
          if (window.__beautyDisposePreview) window.__beautyDisposePreview();
          const box = document.createElement('section'); box.dir = 'rtl';
          const title = document.createElement('p');
          title.textContent = 'از چپ: عکس اصلی، چهرهٔ هم‌تراز، مرجع مو، نتیجه';
          const image = document.createElement('img');
          image.style.cssText = 'display:block;max-width:100%;height:auto';
          image.alt = 'مقایسهٔ عکس ورودی و نتیجهٔ انتقال مو';
          const close = document.createElement('button');
          close.textContent = 'بستن و پاک‌کردن پیش‌نمایش';
          let text = atob(PAYLOAD);
          const pixels = Uint8Array.from(text, c => c.charCodeAt(0)); text = '';
          let url = URL.createObjectURL(new Blob([pixels], {type:'image/png'}));
          pixels.fill(0);
          image.src = url; box.append(title, image, close); document.body.append(box);
          let timer;
          const dispose = () => {
            clearTimeout(timer); image.removeAttribute('src');
            if (url) URL.revokeObjectURL(url); url = null; box.remove();
            window.removeEventListener('pagehide', dispose);
            if (window.__beautyDisposePreview === dispose) delete window.__beautyDisposePreview;
          };
          window.__beautyDisposePreview = dispose;
          close.onclick = dispose;
          window.addEventListener('pagehide', dispose, {once:true});
          timer = setTimeout(dispose, 300000);
          return new Promise(resolve => {
            image.onload = () => {image.onload = null; image.onerror = null; resolve(true);};
            image.onerror = () => {dispose(); resolve(false);};
          });
        })()
        '''.replace('PAYLOAD', json.dumps(encoded))
        if eval_js(script, timeout_sec=30) is not True:
            raise RuntimeError('نمایش تصویر در مرورگر موفق نشد؛ دوباره همین سلول را اجرا کن.')
    finally:
        encoded = script = None
        gc.collect()

def run(service='hairstyle', repeats=1, reference_color=False,
        target_rgb=(116, 52, 40), strength=.85):
    """One interactive case; return anonymous metrics, consume all image buffers."""
    if service not in {'hairstyle', 'color'}:
        raise ValueError('Unknown service')
    import runtime
    api = runtime.api()
    face = reference = response = None
    try:
        # Stop another service before loading HairFastGAN, so T4 holds one engine.
        print('۱/۴ — آماده‌سازی موتور ' + service + '؛ این زمان جزو inference نیست.', flush=True)
        if (api.CLIENT is None or api.CLIENT.service != service
                or api.CLIENT.process.poll() is not None):
            api.start_engine(service)
        print('۲/۴ — عکس فرد را از دستگاه خودت انتخاب کن.', flush=True)
        face = pick_image('عکس فرد: یک چهرهٔ واضح، ترجیحاً روبه‌رو')
        if service == 'hairstyle' or reference_color:
            print('۳/۴ — عکس مرجع مو را انتخاب کن.', flush=True)
            reference = pick_image('عکس مرجع مو: فقط یک چهره، موها واضح و قابل مشاهده')
        print('۴/۴ — پردازش عکس‌های انتخاب‌شده…', flush=True)
        # This explicit in-memory path allows ephemeral preview while preserving
        # PRIVATE_IMAGES and suppressing request-time private diagnostics.
        response = api.CLIENT.call({
            'op': 'evaluate', 'face': face, 'reference': reference,
            'repeats': repeats, 'return_images': True, 'debug_public': False,
            'reference_color': reference_color, 'target_rgb': target_rgb, 'strength': strength,
        })
        metrics = response.get('metrics', {})
        if metrics.get('status') != 'ok':
            error = metrics.get('error_code', 'unknown_error')
            explanations = {
                'exactly_one_face_required': 'در هر عکس باید دقیقاً یک چهرهٔ قابل تشخیص باشد.',
                'cuda_out_of_memory': 'حافظهٔ GPU کافی نبود؛ پردازش GPU دیگری را متوقف کن.',
                'image_limits_exceeded': 'عکس باید ثابت و حداکثر ۱۶ میلیون پیکسل باشد.',
                'unsupported_image_format': 'فقط عکس JPEG، PNG یا WEBP ثابت پشتیبانی می‌شود.',
                'invalid_image_bytes': 'فایل عکس معتبر نیست یا از محدودیت حجم عبور کرده است.',
            }
            raise RuntimeError(explanations.get(error, 'پردازش موفق نشد.') + ' [' + str(error) + ']')
        print('زمان خالص: %.3f ثانیه | پایپ‌لاین: %.3f ثانیه | Peak allocated: %.1f MiB'
              % (metrics['inference_s'], metrics['pipeline_s'], metrics['peak_allocated_mib']), flush=True)
        # Erase parent inputs and the separate result before rendering comparison.
        api.clear_buffer(face); api.clear_buffer(reference)
        api.clear_buffer(response.get('output_png'))
        show_transient(response['comparison_png'])
        print('پیش‌نمایش آماده است؛ پس از ۵ دقیقه بسته می‌شود. این تست تک‌عکس است، نه گیت ۳۰ عکس.', flush=True)
        return dict(metrics)
    except (ValueError, RuntimeError) as exc:
        print('متوقف شد: ' + str(exc), flush=True)
        return {'status': 'error', 'error_code': type(exc).__name__}
    finally:
        api.clear_buffer(face); api.clear_buffer(reference)
        if response is not None:
            api.clear_buffer(response.get('output_png'))
            api.clear_buffer(response.get('comparison_png'))
            response.clear()
        face = reference = response = None
        api.stop_engine()
        gc.collect()
