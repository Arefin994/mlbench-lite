# Contributing to mlbench-lite

Thanks for taking the time to contribute.

## Reporting bugs

Open an issue on GitHub with:

- Python version and OS
- Installed optional extras (`boosting`, `deep`, `tui`)
- A minimal code snippet or CLI command that reproduces the problem
- The full error message or unexpected output

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Install the development environment:

   ```bash
   pip install -e ".[dev]"
   ```

3. Make your changes. Match the existing code style.
4. Run the test suite:

   ```bash
   python -m pytest tests/ -v
   ```

5. Open a pull request against `main`. Describe what changed and why.

## Scope

`mlbench-lite` targets tabular `(X, y)` benchmarking. Keep changes focused on that scope. New model backends should follow the existing pattern in `mlbench_lite/benchmark.py` and degrade gracefully when optional dependencies are missing.

## Code of conduct

Be direct and respectful in issues and pull request reviews.
