# Contributing

Thanks for your interest in this project. It follows standard Python
open-source conventions.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Before opening a pull request

Please make sure the checks that CI runs pass locally:

```bash
make format   # black + ruff --fix
make lint     # ruff + mypy
make test     # pytest
```

## Conventions

- **Style:** [black](https://github.com/psf/black) (line length 100) and
  [ruff](https://github.com/astral-sh/ruff) for linting.
- **Typing:** public functions are type-annotated; `mypy` runs in CI.
- **Tests:** new functionality should come with unit tests under `tests/`.
- **Reproducibility:** anything stochastic must respect the global seed
  (`political_sentiment.utils.seed_everything`).
- **Commits:** small, focused commits with clear messages.

## Adding a new model

1. Subclass `political_sentiment.models.base.SentimentModel`.
2. Register it in `political_sentiment.models.registry.build_model`.
3. Add a config under `configs/`.
4. Add tests (mirror `tests/test_models_baseline.py`).
