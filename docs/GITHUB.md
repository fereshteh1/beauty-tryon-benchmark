# Publishing this project

Create a public repository named `beauty-tryon-benchmark` on the maintainer account `fereshteh1`. Publish the **contents** of the project folder, including `.github`, `.gitignore` and `.gitattributes`; uploading only the ZIP is less useful for browsing/testing. The standalone notebook can be opened through Colab's GitHub tab using the repository URL after publication. The initial maintainer is Fereshteh Abolghasemi (`fereshteh1`). Until the repository is actually created, no final GitHub repository URL is claimed in this package.

Recommended description: `Phase-zero salon hair try-on benchmark: HairFastGAN, SegFormer + LAB, Colab, latency/VRAM and in-memory photo processing.`

Before publication, install CPU test dependencies, run tests, regenerate/validate the notebook if sources changed, and review licenses. The distributed notebook has no execution outputs and contains no customer images. `.gitignore` protects common generated paths; it does not prevent intentionally adding sensitive files with `git add -f`.

Suggested initial commit message: `Add phase-zero hair try-on benchmark and Colab notebook`. Use meaningful future commits for bug fixes, documented measurements and reviewed changes. Do not manufacture stars, contributors or a commit history. A public repository alone does not establish eligibility for any unspecified developer offer.

Keep model authorship explicit, historical demo measurements labeled, and the incomplete 30-image gate visible. Do not claim a finished commercial salon engine or commercial rights to all third-party components.

## Ownership checklist before first public push

- Keep the root MIT `LICENSE` notice as `Copyright (c) 2026 Fereshteh Abolghasemi`.
- Keep `COPYRIGHT.md` and `THIRD_PARTY_NOTICES.md` in the root.
- Do not remove upstream copyright/license notices from third-party code.
- Do not describe HairFastGAN, SEAN, CLIP, SegFormer or their weights as owned by this repository.
- The linked Colab run may be cited as feasibility evidence, but do not claim that it completes the pending 30-image quality gate.
