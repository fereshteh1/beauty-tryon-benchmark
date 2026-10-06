# Beauty Try-On Benchmark

[![CPU tests and release validation](https://github.com/fereshteh1/beauty-tryon-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/fereshteh1/beauty-tryon-benchmark/actions/workflows/ci.yml)

Phase-zero feasibility and benchmarking tools for salon hair consultation. Give a target portrait and a hair reference to the Colab notebook, receive a visual approximation, and measure latency and GPU memory. This repository contributes orchestration, LAB recoloring, privacy controls, tests and a benchmark protocol around existing research models. It does not train or claim authorship of HairFastGAN.

[Open the validated Colab prototype](https://colab.research.google.com/drive/1B8Mj0FYPIQ4B9N0EQ2L5t2udpY1YKp4t?usp=sharing) · [راهنمای فارسی](README_FA.md) · [Colab instructions](docs/COLAB.md) · [Benchmark protocol](docs/BENCHMARK.md) · [Privacy](docs/PRIVACY.md) · [Copyright](COPYRIGHT.md) · [Licenses](THIRD_PARTY_NOTICES.md)

## Project status

**v0.1.0 is a public phase-zero pre-release.** CPU CI is passing, a Tesla T4 feasibility run is documented, and the reproducible benchmark harness is available now. The complete real 30-image quality gate is still pending and is tracked as the next validation milestone.


## What is included

| Engine | Implementation | Phase-zero target |
|---|---|---|
| Hairstyle and reference color | HairFastGAN, face alignment, FP32, batch size 1 | At least 21/30 acceptable outputs with inference <8 s on T4/L4 |
| Hair recoloring | SegFormer-B0 hair mask + LAB chroma transform | Independent inference target <0.5 s |

The hairstyle engine produces an aligned 1024×1024 face crop. Its reference supplies both hairstyle and hair color. Recoloring preserves LAB L* before conversion to RGB; gamut clipping can alter final RGB luminance. It does not simulate bleaching or produce reliably light blond from very dark hair.

StableHair v2, PSGAN/SCGAN and FaceMesh makeup presets are not implemented. No mobile application, product UI, database, beauty scoring or physical hair simulation is included.

## Run in Colab

1. Open `notebooks/Beauty_TryOn_Phase0_T4.ipynb` in Google Colab using **File → Upload notebook**.
2. Select **Runtime → Change runtime type → T4 GPU** and run all cells in order.
3. Installation is cell **4**, assets are cell **6**, CUDA/tests are cell **16**, public demos are **17–18**, and the current-run report is **19**.
4. Cell **20 is enabled by default**. Choose the target portrait, then the hair reference. Results appear in a temporary comparison, with inference, pipeline and peak allocated memory metrics.
5. For another pair, rerun only **20**. Cell **21** optionally recolors from a reference; cell **22** optionally benchmarks a dataset ZIP.

Each image must have exactly one detectable face, be a static JPEG/PNG/WEBP, be at most 25 MiB and contain at most 16 million pixels. The maximum decoded side is 1600 pixels. File selection times out after about five minutes; rerun the photo cell if it expires.

The notebook contains all source code in separate visible cells. It does not need this repository uploaded to Colab. First installation and dlib compilation can take time. After a fresh Colab runtime, model files and environment must be recreated; saved cell outputs do not restore runtime files.

## Linux CUDA commands

The intended GPU runtime is Python 3.10 with PyTorch 2.2.2/cu121 in an isolated environment. The setup command creates that environment. A Linux NVIDIA driver, CUDA GPU, git, internet access and roughly 18 GiB free disk are required. Installing missing build tools can require system administrator privileges; Google Colab provides these permissions.

```bash
python3 scripts/beauty_cli.py --root /content/beauty_phase0 setup
python3 scripts/beauty_cli.py --root /content/beauty_phase0 status
python3 scripts/beauty_cli.py --root /content/beauty_phase0 demo --service hairstyle
python3 scripts/beauty_cli.py --root /content/beauty_phase0 demo --service color
```

Use `--backend torch_reference` before the subcommand for a compiler-independent operator baseline; record it alongside results. The default `auto` attempts the original CUDA operators and records any reference fallback. Startup/warmup are not inference measurements.

Public/synthetic datasets already on disk can be processed sequentially and exported:

```bash
python3 scripts/beauty_cli.py --root /content/beauty_phase0 batch-public /path/to/public-dataset --public-data --save-outputs
```

CLI demos save public comparisons under `public_demo_outputs`, without requiring an IPython frontend. With `--save-outputs`, a public batch writes each comparison before asking for human scores; open that printed path to review it. Without export, the CLI batch produces metrics without human scores and remains UNASSESSED. Image export and folder mode are restricted to public/synthetic data permitted for this use. Private photo batches use the Colab memory ZIP path, not a folder upload. See [BENCHMARK.md](docs/BENCHMARK.md).

## Validation status

A linked Colab prototype run on **2026-10-05** completed on a **Tesla T4** with Python 3.10 and PyTorch 2.2.2+cu121. The saved run shows CUDA arithmetic success, **12/12 core tests**, **9/9 runtime tests**, the pinned six-operator check, and successful public demos for both engines:

| Engine | GPU | Inference | Pipeline | Peak CUDA allocated |
|---|---|---:|---:|---:|
| HairFastGAN | Tesla T4 | 2.061 s | 10.242 s | 6905.4 MiB |
| SegFormer + LAB | Tesla T4 | 0.142 s | 0.151 s | 305.2 MiB |

These measurements are **single-demo observations from the linked professional Colab prototype**, not the complete 30-image benchmark and not proof of the phase-zero quality gate. The repository notebook/source package is maintained separately, so the linked prototype run is evidence of feasibility rather than a claim that every repository revision has been GPU-validated byte-for-byte. The complete 30-image quality gate remains pending. Details are retained in `reports/observed_demo.json`.

## Development

```bash
python3 -m pip install -r requirements-cpu.txt
python3 -m unittest discover -s tests -v
python3 scripts/validate_release.py
node tests/browser_lifecycle.cjs
```

CPU CI does not download models or run inference. The StyleGAN operator test is skipped without `BEAUTY_TEST_REPO`; `beauty_cli.py verify` enables it after GPU setup. The actual downloaded demo PNG test is skipped when assets are absent; it is enabled during GPU verification. Test fixtures are synthetic and are not model-quality results.

After changing engine sources, regenerate the standalone notebook with `python3 scripts/build_notebook.py`. Its visible `%%writefile` cells must agree with the source files.

## Copyright, licensing and contribution

Copyright in the repository's original 2026 code and documentation is held by **Fereshteh Abolghasemi**. Publishing the project under MIT grants broad reuse rights but does **not** transfer that copyright. See [COPYRIGHT.md](COPYRIGHT.md) and the root [LICENSE](LICENSE).

Downloaded models, weights, datasets and third-party source retain their own terms. HairFastGAN declares MIT; SEAN declares CC BY-NC-SA 4.0 and academic research use only. This repository does **not** grant blanket commercial permission for the combined pipeline. Consult [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before reuse.

Contributions should improve reproducibility, fix meaningful failures, document limitations or add measured benchmarks. Keep customer images, credentials, virtual environments, weights and generated private previews out of the repository. See [CONTRIBUTING.md](CONTRIBUTING.md).
