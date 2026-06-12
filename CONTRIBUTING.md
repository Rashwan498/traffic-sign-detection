# Contributing

Thanks for your interest in contributing to VisionProject_handoff!

How to contribute

1. Fork the repository and create a feature branch: `git checkout -b feature/your-change`.
2. Write clear commit messages and keep changes focused.
3. Run tests and linters locally before making a PR (see below).
4. Open a pull request against `main`

Code style and linters

- We follow standard Python formatting. Please run `flake8` and fix reported issues.

Running tests

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest flake8
```

Run tests and linters:

```bash
flake8 --max-line-length=88
pytest -q
```

Reproducing experiments

See the repository `README.md` for Quickstart and exact commands used to reproduce training and evaluation runs.

License

By contributing you agree to license your contributions under the MIT License (see `LICENSE`).
