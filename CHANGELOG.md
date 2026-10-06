# Changelog

## 0.1.0 — 2026-10-05

- Source-only phase-zero project with a standalone 23-cell Colab notebook.
- HairFastGAN and independent SegFormer + LAB engines with pinned assets.
- In-memory portrait/reference and ZIP inputs, private transient comparisons and human scoring.
- CUDA/NVML metrics, anonymous exports and separate quality/latency gates.
- CLIP build isolation and PNG buffer ownership fixes integrated from the working prototype.
- CPU, IPC, privacy and browser lifecycle tests; GPU operator checks after setup.
- Publicly shared 2026-10-05 Tesla T4 prototype run documented as feasibility evidence, including successful tests and measured single-demo latency/VRAM; the 30-image quality gate remains pending.
- Initial copyright holder clarified as Fereshteh Abolghasemi, with explicit third-party ownership boundaries and an added `COPYRIGHT.md`.
- Image cleanup made idempotent across newer Pillow releases without decoding rejected lazy images.
