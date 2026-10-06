# Contributing

Use a Linux/Colab GPU environment for inference and the lightweight CPU requirements for harness tests. Run `python3 -m unittest discover -s tests -v`, regenerate the notebook with `scripts/build_notebook.py` when engine sources change, and run `scripts/validate_release.py`. Browser lifecycle fixtures run with `node tests/browser_lifecycle.cjs`.

Report a concrete trigger, anonymous error code, dependency/backend versions and GPU. Use synthetic/public fixtures with appropriate rights. Do not attach customer photos, face embeddings, credentials or private notebook outputs to issues or pull requests.

Benchmark contributions must identify dataset groups, metric definitions, repetitions, actual backend and failure counts. Label partial/unrated runs. Do not claim phase-zero PASS until the complete agreed 30-image gate is met. Fixes to model APIs must credit upstream work and retain its licenses.

## Contribution licensing

By submitting a contribution for inclusion in this repository, you agree that your contribution may be distributed under the repository's MIT License. Unless a separate written agreement says otherwise, contributors retain copyright in their own contributions; no copyright assignment to the project owner is implied.
