# Image lifecycle

The original constraints concern customer images: strip EXIF/GPS before inference, process images in memory, avoid disk/database/object storage, and clean buffers after output generation. Code, compiled operators, dependency environments, public weights and public warmup assets may be stored on disk.

1. The Colab picker reads a browser-selected file through FileReader and returns base64 over a Colab JavaScript RPC. It does not use `google.colab.files.upload()`, which creates files.
2. The parent owns mutable bytearrays. An isolated worker receives framed local IPC, applies EXIF orientation, validates size/format, and creates a fresh RGB image without metadata before passing it to models.
3. An audit hook blocks file writes, subprocesses and network connections during customer requests. Public model initialization is separate and may fetch public backbone weights. This is a Python-side side-effect guard, not an OS sandbox or proof that arbitrary native extensions have no side effects.
4. Inputs, PIL/NumPy tensors and known upstream retained image state are cleared on success/error. Owned encoded buffers are zeroed before close, including PNG early-close cases. The worker is closed after each single preview; each batch keeps one model resident and clears per-case state.
5. Private pixels are returned only to an explicitly chosen transient comparison callback. `IPython.display.Image` and ordinary image output/export remain blocked in private mode. Browser previews use Blob URLs, expire after five minutes, can be closed manually and replace the previous preview. URL revocation is resource cleanup, not secure erasure of browser or compositor memory.
6. Batch ZIP bytes are held in memory and never extracted. Anonymous reports store ID/group/scores/status/timing/memory, not names, source image paths, face embeddings or pixel arrays. Private folder mode and image export are blocked.

## Limits

Python, Pillow, pickle, base64, CUDA libraries and browsers create immutable or hidden copies. Overwriting owned bytearrays/tensors is **best effort**, not guaranteed secure wiping. GPU driver caches, Jupyter transport, Google Colab infrastructure and provider retention are outside the harness's control. RPC transport avoids standard saved image outputs, but this repository does not independently audit the Colab frontend or promise that Google never retains data. No encrypted/private production backend is claimed.

Original photos already exist on the uploader's device; the pipeline does not erase those originals. Public demo frames intentionally use normal notebook output and may be retained in saved notebook history. Never use `PUBLIC_IMAGES=True` for customer data merely to bypass private display guards. Do not publish customer photos or private notebook outputs to GitHub.

The release validator rejects bundled photos/weights, nonempty notebook outputs, obvious credential patterns and unsynchronized source cells. It is a release check, not a comprehensive data-loss prevention system. Review any new material before committing it.
