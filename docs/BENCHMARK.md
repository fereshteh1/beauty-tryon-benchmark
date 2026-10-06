# Phase-zero benchmark protocol

## Dataset

Use 30 target portraits, one anonymous ID per target, and an explicit reference for hairstyle cases. Group A: 10 well-lit, frontal, unobstructed salon portraits. Group B: 10 portraits at approximately 30–45 degrees with warm lighting, curly/frizzy hair, glasses or strands on the forehead. Group C: 10 direct mobile photos under real salon conditions. A group label is an evaluator declaration, not something the code can verify from the image.

`examples/manifest.json` has all 30 required IDs and path placeholders. It contains no photographs; replace paths with actual members of your local ZIP. Keep `manifest.json` at the archive root. A ZIP can contain `targets/A01.jpg` and `references/style01.jpg`. Do not give images people's names. The same reference may be reused; ZIP member bytes are zeroed only after their final use.

Partial batches are useful for debugging but produce INCOMPLETE. Duplicate IDs, invalid relative paths or missing members are rejected. ZIP size is limited to 256 MiB at selection, 512 MiB total uncompressed, 256 entries, 25 MiB per entry and compression ratio ≤200. Encrypted ZIPs are rejected. It is never extracted by the pipeline.

## Ratings

Scores are human observations; the harness never infers quality from speed.

| Score | Meaning |
|---|---|
| 1 | Unusable |
| 2 | Major defects |
| 3 | Visible problems |
| 4 | Acceptable for an approximate consultation preview |
| 5 | Excellent |

Rate identity preservation, natural hairline and requested reference fidelity. Compare the original portrait as well as the aligned target, so alignment does not conceal changes to identity. Do not assess facial attractiveness. Agree on the proposed operational definition “all three scores ≥4” before collecting final results. Record conditions or failures in an anonymous review log when necessary.

Private Colab batches use a transient comparison plus a three-integer prompt. Human scores are anonymous metadata; they can be exported without image pixels. If review is disabled, successful cases remain unrated and the gate is UNASSESSED. Scores supplied through a callback are still required to be integers 1–5, not booleans.

## Timing and memory

- `inference_s`: measured around the model's pure inference path, with CUDA synchronization; three repetitions and their median for batches/demos. The interactive preview uses one repetition by default.
- `alignment_s`: face alignment/preparation measured separately; zero for recolor alignment.
- `pipeline_s`: necessary processing from preparation through image production, including alignment and tensor-to-image conversion. Upload, input decode, comparison rendering, model loading and warmup are reported/excluded separately.
- `decode_s`: decode, EXIF orientation and creation of clean RGB input.
- `provenance.startup.model_load_s` / `warmup_s`: initialization and public warmup, separately reported in readiness metadata. First initialization can include public backbone downloads. `init_peak_allocated_mib` / `init_peak_reserved_mib` cover model loading before warmup and are separate from inference peaks.
- Peak allocated/reserved: CUDA allocator maximum; resident weights are included. Startup memory is not this inference metric.
- NVML process/device peak: sampled every 10 ms; shorter spikes may be missed. Missing NVML readings remain null.

Record GPU, source/weight/model revisions, resolved requirements checksum, requested/actual operator backend, FP32 profile, seed 3407, and repetitions. CUDA fused and PyTorch reference runs should be compared as separate backend profiles. The seed helps comparison but does not guarantee bitwise determinism across drivers or kernels.

## Gate

For HairFastGAN require the complete 30-ID protocol, at least 21 quality passes, and at least 21 **joint** quality passes with `inference_s < 8`. Boundary 8.000 fails the latency condition. Failed cases stay in the denominator. Missing human ratings or valid latency give UNASSESSED; an incomplete/mixed-engine dataset gives INCOMPLETE; otherwise the outcome is PASS or FAIL.

P95 is reported only; it is not an extra acceptance criterion. Color uses an independent exclusive limit of 0.5 seconds and its own results. Do not pool engines. For color reference fidelity, document the chosen fixed RGB preset, or use the separate interactive reference-color test; folder/ZIP color batches default to the fixed preset.

Observed single-demo speed is not proof of robustness or complete phase-zero approval. The final report must include real outcomes, failures, scores and the quality/latency counts. Never insert synthetic fixture results into a model benchmark report.
