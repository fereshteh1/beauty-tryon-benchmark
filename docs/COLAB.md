# Colab run and recovery

Validated professional prototype: https://colab.research.google.com/drive/1B8Mj0FYPIQ4B9N0EQ2L5t2udpY1YKp4t?usp=sharing

The linked professional prototype is maintained separately from the repository-generated standalone notebook. Its saved 2026-10-05 T4 run is feasibility evidence; it is not a byte-for-byte validation claim for every repository revision.

Open the standalone notebook, choose T4 GPU, and run cells in order. The notebook has 23 numbered code cells with readable source. `Writing ...` from a source-definition cell means the program source was written; it does not mean inference succeeded. Setup code goes to `/content/beauty_phase0/benchmark_code/engine` with tests in the sibling `tests` folder. Environments and public model assets live under `/content/beauty_phase0`.

| Condition | Action |
|---|---|
| `pkg_resources`/CLIP build failure | This release pins setuptools/wheel and disables build isolation for CLIP. Rerun cell 4; keep the actual installer diagnostic if it still fails. |
| Runtime/model files missing | Recreate the setup in cells 1–16. Saved notebook outputs do not restore runtime files. |
| GPU missing | Select T4 in runtime settings, then rerun preparation in the resulting runtime. |
| `I/O operation on closed file` for PNG | The release includes the ownership/close fix in cell 7. Use this complete notebook rather than old patch cells. |
| Exactly one face required | Choose a portrait/reference with one clear detectable face; rerun 20. |
| File selection cancelled/timed out | Rerun only 20 (or the relevant optional cell). |
| Inference fails | The cell reports an anonymous error. Keep GPU/init diagnostics, avoid sending customer photos in bug reports. |
| Startup appears slow | First initialization can fetch public torchvision/CLIP backbone caches; that is not inference latency. |

Public demos are cells 17 and 18; cell 19 rejects stale/incomplete demo status. Cell 20 requests user images by default. To skip it for an unattended public-demo run, set `RUN_SINGLE=False` before Run all. Optional flags in 21–22 control ordinary benchmark modes, not fixes to program code.

For another pair of photos, rerun 20; the model loads from already-present assets, and installation/download cells need not be rerun. For a 30-case private batch, set `RUN_BATCH=True` in 22, upload a prepared ZIP through the memory picker, and provide three quality scores per successful case. Results are metrics only; temporary browser comparisons replace one another.
