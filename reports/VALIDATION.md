# Delivery validation — 2026-10-05

Final source-package verification was run on Linux with Python 3.13.5, NumPy 2.3.5, Pillow 12.3.0 and OpenCV 4.13.0. The repository's `requirements-cpu.txt` still pins the intended lightweight CPU test environment separately. GPU inference is intended for the isolated Python 3.10/PyTorch 2.2.2/cu121 environment created by setup.

- Unittest suite: **35 tests, 33 passed, 2 skipped** in the final packaging environment. One skip requires downloaded public demo assets; the other requires the downloaded GPU/StyleGAN operator assets.
- The image cleanup path was made idempotent across newer Pillow versions without forcing decode of a rejected lazy image; JPEG/PNG/WEBP cleanup, EXIF handling, limits and buffer ownership tests pass.
- Transient private preview, cancellation, output cleanup, public-export-before-review and a synthetic 30-case gate test passed. Synthetic fixture scores/timings are not model benchmark measurements.
- A fresh subprocess deployed the notebook's actual source cells, assembled the worker and imported the CPU-safe APIs with no old kernel globals or folder assumptions.
- Browser DOM fixtures: four picker paths and four preview lifecycle paths passed, including expiry, replacement and failed image load. This is a lifecycle unit test, not an audit of the live Colab frontend.
- `scripts/validate_release.py` passed: notebook schema, Python syntax, 23 code cells, source parity, empty outputs, 30-ID manifest template and source-only inventory are synchronized.

A separate, publicly shared professional Colab prototype was inspected on 2026-10-05. Its saved outputs show **Tesla T4** execution, CUDA arithmetic success, **12/12 core tests**, **9/9 runtime tests**, the pinned operator test, HairFastGAN inference **2.061 s** / pipeline **10.242 s** / **6905.4 MiB** peak allocated, and SegFormer+LAB inference **0.142 s** / pipeline **0.151 s** / **305.2 MiB** peak allocated. These are recorded in `observed_demo.json`.

That linked prototype run is **feasibility evidence**, not a byte-for-byte GPU validation claim for this repository release. The complete real 30-image quality gate remains **not established**. Published CPU tests and the linked single-demo run do not replace that gate.
