"""Generate the standalone Colab notebook from the visible repository sources."""
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = '/content/beauty_phase0/benchmark_code/engine'
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(source):
    ast.parse(source.split('\n', 1)[1] if source.startswith('%%writefile ') else source)
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': source.splitlines(keepends=True),
                  'execution_count': None, 'outputs': []})


def section(number, title, detail, source=None, file=None):
    markdown(f'## سلول {number} — {title}\n\n{detail}\n')
    if file:
        target = str(Path(CODE).parent / file) if file.startswith('tests/') else CODE+'/'+Path(file).name
        code('%%writefile '+target+'\n'+(REPO/file).read_text())
    else:
        code(source)


LOAD = f'''import os, pathlib, sys
_code = pathlib.Path('{CODE}')
if not (_code/'runtime.py').is_file():
    raise RuntimeError('ابتدا سلول‌های ۱ و ۲ را اجرا کن؛ کد پایه در این Runtime موجود نیست.')
os.environ['BEAUTY_ROOT'] = '/content/beauty_phase0'
os.environ['BEAUTY_TEST_ROOT'] = str(_code.parent/'tests')
if str(_code) not in sys.path:
    sys.path.insert(0, str(_code))
import runtime as ctx
'''

markdown('''# Beauty Try-On Benchmark — فاز صفر / نسخهٔ 0.1.0

این نوت‌بوک دو موتور را اجرا می‌کند: انتقال مدل و رنگ مو با **HairFastGAN**، و رنگ مو با **SegFormer-B0 + LAB**. مدل‌های پایه متعلق به سازندگان اصلی‌اند؛ سهم این پروژه اجرای Colab، ارزیابی، سنجه‌ها، کنترل پردازش تصویر و بنچمارک است.

**روش اجرا:** Runtime → Change runtime type → T4 GPU → Save، سپس Run all.
۲۳ سلول شماره‌دار و خوانا داریم. سلول‌های `%%writefile` فقط کد می‌نویسند؛ فایل عکس مشتری ایجاد نمی‌کنند. نصب در ۴، وزن‌ها در ۶، سلامت در ۱۶، دموها در ۱۷ و ۱۸ و گزارش در ۱۹ است. **سلول ۲۰ به‌طور پیش‌فرض روشن است و عکس فرد و مرجع را از تو می‌گیرد.** اگر مهلت انتخاب عکس تمام شد، فقط سلول ۲۰ را دوباره اجرا کن.

تست اختیاری رنگ روی عکس تو در ۲۱ و بنچمارک ZIP در ۲۲ هستند و پیش‌فرض خاموش‌اند. پس از اجرای اولیه، برای عکس تازه فقط ۲۰ را اجرا کن؛ نصب را تکرار نکن. اگر Colab محیط اجرای جدیدی ساخته باشد، کد نوت‌بوک و خروجی‌های قبلی فایل‌های نصب را بازنمی‌گردانند و باید مراحل آماده‌سازی دوباره اجرا شوند.

**حریم خصوصی:** دریافت با FileReader در حافظه، پاکسازی EXIF/GPS، عدم استخراج ZIP و عدم ذخیرهٔ عکس خصوصی. نمایش عکس خصوصی با RPC موقت و Blob مرورگر است؛ بعد از ۵ دقیقه، بستن پیش‌نمایش یا نمایش نتیجهٔ بعدی پاک می‌شود. نمایش به خروجی استاندارد تصویری نوت‌بوک ارسال نمی‌شود. بااین‌حال پاکسازی امن همهٔ نسخه‌های پنهان Python/CUDA/مرورگر و نگهداری داده توسط Google قابل تضمین یا ممیزی مستقل این کد نیست. قاب دموهای عمومی ممکن است در خروجی ذخیره‌شدهٔ نوت‌بوک بماند.

**وضعیت:** یک نمونهٔ حرفه‌ای جداگانه در Colab در ۵ اکتبر ۲۰۲۶ روی Tesla T4 با موفقیت اجرا و سنجه‌های تک‌دمو ثبت شده است؛ این موضوع فقط مدرک feasibility است و به معنی GPU-validation بایت‌به‌بایت این revision نیست. آزمون کامل ۳۰ عکس و گیت کیفیت هنوز تکمیل نشده‌اند. این نوت‌بوک محصول نهایی سالن، اپ موبایل یا پنل کاربری نیست. محدودیت‌های SEAN و مجوزهای مدل‌ها را پیش از استفادهٔ تجاری بررسی کنید.

**کپی‌رایت:** کد و مستندات اصلی این پروژه © 2026 Fereshteh Abolghasemi و تحت MIT منتشر می‌شوند؛ مدل‌ها و وابستگی‌های شخص ثالث متعلق به صاحبان خودشان هستند.
''')

section(1, 'بررسی GPU و آماده‌سازی مسیر', 'شروع یک اجرای جدید؛ هیچ عکس خصوصی دریافت نمی‌شود.', f'''import os, pathlib, shutil, subprocess, sys
_root = pathlib.Path('/content/beauty_phase0')
_code = pathlib.Path('{CODE}')
_code.mkdir(parents=True, exist_ok=True)
(_code.parent/'tests').mkdir(parents=True, exist_ok=True)
for name in ('_beauty_benchmark_api', '_beauty_professional_v2_api', '_beauty_professional_readable_api'):
    previous = sys.modules.get(name)
    if previous is not None:
        previous.stop_engine()
    sys.modules.pop(name, None)
for name in ('runtime', 'controller', 'photo_preview'):
    sys.modules.pop(name, None)
os.environ['BEAUTY_ROOT'] = str(_root)
os.environ['BEAUTY_TEST_ROOT'] = str(_code.parent/'tests')
(_code/'run_status.json').write_text('{{}}')
if not shutil.which('nvidia-smi'):
    raise RuntimeError('Runtime → Change runtime type → T4 GPU را انتخاب کن.')
subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'], check=True)
if shutil.disk_usage(_root).free < 18 * 2**30 and not (_root/'weight_manifest.json').exists():
    raise RuntimeError('نصب اول حداقل ۱۸ GiB فضای آزاد لازم دارد.')
print('GPU بررسی شد؛ مراحل بعد کد و وزن‌های عمومی را آماده می‌کنند.')
''')
section(2, 'تنظیمات و مدیریت اجرا', 'مسیرها، مدیریت مدل و وضعیت مراحل؛ فقط نوشتن کد.', file='engine/runtime.py')
section(3, 'کد نصب مستقل', 'Python 3.10، نسخه‌های ثابت، CLIP و dlib؛ نصب در سلول بعد است.', file='engine/install.py')
section(4, 'نصب کتابخانه‌ها', 'نصب اول ممکن است برای کامپایل dlib زمان‌بر باشد؛ منتظر پایان همین سلول بمان.', LOAD+"\nwith ctx.stage(4, 'نصب کتابخانه‌ها'):\n    ctx.install()\n")
section(5, 'کد دریافت وزن‌ها و اصلاح اپراتورها', 'نسخه‌های ثابت و fallback ثبت‌شونده؛ هیچ وزن یا عکس خصوصی داخل خود نوت‌بوک نیست.', file='engine/assets.py')
section(6, 'دریافت وزن‌های عمومی', '۱۶ وزن HairFastGAN، سگمنتیشن و نمونه‌های عمومی گرم‌سازی دریافت می‌شوند.', LOAD+"\nwith ctx.stage(6, 'دریافت وزن‌ها'):\n    ctx.assets()\n")
section(7, 'بافر تصویر، EXIF و LAB', 'اصلاح بسته‌شدن بافر PNG از ابتدا اعمال شده است.', file='engine/core.py')
section(8, 'موتور انتقال مدل مو', 'چهره و رفرنس باید هرکدام دقیقاً یک چهرهٔ قابل تشخیص داشته باشند.', file='engine/hair_engine.py')
section(9, 'موتور رنگ مستقل', 'سگمنتیشن و تغییر a*/b* در LAB؛ روشن‌کردن شدید موی تیره پوشش داده نمی‌شود.', file='engine/color_engine.py')
section(10, 'پردازشگر، زمان و VRAM', 'Warmup عمومی، IPC حافظه‌ای و محافظ نوشتن/شبکه در درخواست.', file='engine/worker_tail.py')
section(11, 'ارزیابی و بنچمارک', 'ZIP بدون استخراج، ارزیابی انسانی، گزارش بی‌نام و گیت.', file='engine/controller.py')
section(12, 'دریافت عکس و نمایش موقت', 'بدنهٔ کامل FileReader و پیش‌نمایش موقت؛ تصاویر در فایل نوشته نمی‌شوند.', file='engine/photo_preview.py')
section(13, 'تست‌های تصویر و گیت', 'JPEG/PNG/WEBP، متادیتا، ماسک، ZIP، محدودیت‌ها و گیت.', file='tests/test_core.py')
section(14, 'تست ارتباط و Runtime', 'Timeout، خطای نصب، پاکسازی و جلوگیری از نتیجهٔ قدیمی.', file='tests/test_runtime.py')
section(15, 'تست شش اپراتور', 'در سلامت GPU با assets واقعی اجرا می‌شود؛ CI بدون مدل آن را skip می‌کند.', file='tests/test_operators.py')
section(16, 'ساخت worker و اجرای سلامت', 'محاسبهٔ CUDA و ۲۶ تست اولیه؛ این مرحله جای استنتاج مدل را نمی‌گیرد.', LOAD+'''\nimport ast
worker = (ctx.CODE_ROOT/'core.py').read_text()+'\n\n'+(ctx.CODE_ROOT/'worker_tail.py').read_text()
ast.parse(worker)
(ctx.CODE_ROOT/'worker.py').write_text(worker)
with ctx.stage(16, 'سلامت CUDA و تست‌ها'):
    ctx.verify()
''')
section(17, 'دموی عمومی HairFastGAN', 'دو warmup و سه تکرار زمان‌سنجی؛ نتیجهٔ crop هم‌تراز ۱۰۲۴×۱۰۲۴ است.', LOAD+"\nwith ctx.stage(17, 'دموی HairFastGAN'):\n    ctx.demo('hairstyle')\n")
section(18, 'دموی عمومی رنگ مو', 'موتور مستقل SegFormer + LAB؛ زمان‌سنجی با انتقال مدل مو ترکیب نمی‌شود.', LOAD+"\nwith ctx.stage(18, 'دموی رنگ مو'):\n    ctx.demo('color')\n")
section(19, 'گزارش همین اجرا', 'فقط دموهای موفق همین اجرا گزارش می‌شوند؛ گیت ۳۰ عکس هنوز جداست.', LOAD+"\nwith ctx.stage(19, 'گزارش دمو'):\n    ctx.summary()\n")
section(20, 'عکس فرد + مرجع مدل مو', 'روشن است. ابتدا عکس فرد و سپس مرجع را انتخاب کن. برای عکس تازه فقط همین سلول را دوباره اجرا کن؛ نصب تکرار نمی‌شود.', LOAD+'''\nRUN_SINGLE = True
if RUN_SINGLE:
    import photo_preview
    single_metrics = photo_preview.run(service='hairstyle', repeats=1)
else:
    print('تست تک‌عکس غیرفعال است.')
''')
section(21, 'تست اختیاری رنگ از مرجع', 'برای تست، RUN_COLOR را True کن؛ فرم مو تغییر نمی‌کند و رنگ مرجع استخراج می‌شود.', LOAD+'''\nRUN_COLOR = False
if RUN_COLOR:
    import photo_preview
    color_metrics = photo_preview.run(service='color', repeats=1, reference_color=True)
else:
    print('تست رنگ از مرجع غیرفعال است.')
''')
section(22, 'بنچمارک ۳۰ عکس', 'برای اجرا RUN_BATCH را True کن. ZIP با manifest.json و عکس‌ها در حافظه خوانده می‌شود. هر نتیجه موقت نمایش داده شده و سه نمرهٔ انسانی دریافت می‌شود. فایل عکس ذخیره نمی‌شود؛ CSV/JSON بی‌نام ذخیره می‌شوند.', LOAD+'''\nRUN_BATCH = False
SERVICE = 'hairstyle'
if RUN_BATCH:
    with ctx.stage(22, 'بنچمارک ۳۰ عکس'):
        batch_gate = ctx.batch(service=SERVICE, public=False, review=True)
else:
    print('بنچمارک ۳۰ عکس هنوز اجرا نشده است.')
''')
section(23, 'آزادسازی منابع', 'پردازش مدل بسته می‌شود. تصویر موقت مرورگر تا بستن یا پایان مهلت قابل مشاهده است.', LOAD+"\nctx.stop()\nprint('پردازش مدل بسته شد؛ برای عکس جدید فقط سلول ۲۰ را اجرا کن.')\n")
markdown('''## معیارها و محدوده

گروه‌های A/B/C هرکدام ۱۰ عکس؛ شناسه‌های A01..A10، B01..B10، C01..C10. گیت HairFastGAN: هر سه نمرهٔ Identity/Hairline/Reference حداقل ۴، حداقل ۲۱ خروجی با کیفیت قابل قبول و حداقل ۲۱ خروجی که همان کیفیت و `inference_s < 8` را همزمان دارند. آستانهٔ ۴ تعریف عملیاتی پیشنهادی است. خطاها در مخرج ۳۰ باقی می‌مانند؛ دیتاست ناقص INCOMPLETE و نمره/زمان ناموجود UNASSESSED است. P95 فقط گزارش می‌شود. موتور رنگ با حد مستقل ۰٫۵ ثانیه ارزیابی می‌شود و نتایج دو موتور ادغام نمی‌شوند.

`inference_s` پردازش خالص، `alignment_s` هم‌ترازی، `pipeline_s` آماده‌سازی لازم تا تولید تصویر، `decode_s` خواندن/پاکسازی ورودی است. بارگذاری مدل و warmup خارج از زمان inference هستند. Peak allocated/reserved از CUDA و process/device از NVML با نمونه‌برداری ۱۰ ms گزارش می‌شوند؛ NVML ممکن است جهش کوتاه‌تر را نبیند.

StableHair v2، PSGAN/SCGAN و آرایش FaceMesh در این نسخه اجرا نشده‌اند. موبایل، UI محصول و دیتابیس تولید نشده‌اند.

منابع: https://github.com/AIRI-Institute/HairFastGAN ؛ https://huggingface.co/AIRI-Institute/HairFastGAN ؛ https://huggingface.co/isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing ؛ https://github.com/openai/CLIP ؛ https://github.com/ZPdesu/SEAN
''')

for index, cell in enumerate(cells):
    cell['id'] = 'beauty-benchmark-%03d' % index
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}}}
destination = REPO/'notebooks/Beauty_TryOn_Phase0_T4.ipynb'
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print(destination, '—', sum(c['cell_type']=='code' for c in cells), 'code cells')
