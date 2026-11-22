"""Configuration loading with base-config inheritance.

Each experiment config may declare its own keys; at load time it is deep-merged
on top of ``configs/base.yaml`` so shared settings (seed, paths, data) live in
one place. The resulting :class:`Config` supports both attribute and dict-style
access and can serialize itself back to YAML for the reproducibility log.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

_DEFAULT_BASE = "configs/base.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """Lightweight recursive namespace over a nested dict.

    Examples
    --------
    >>> cfg = Config({"train": {"lr": 0.001}})
    >>> cfg.train.lr
    0.001
    >>> cfg["seed"] = 7
    >>> cfg.get("missing", 3)
    3
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying nested dict (deep copy)."""
        return copy.deepcopy(self._data)

    def to_yaml(self, path: str | Path) -> None:
        """Persist the resolved config to ``path`` (for the run log)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self._data, handle, sort_keys=False, allow_unicode=True)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Config({self._data!r})"


def load_config(path: str | Path, base_path: str | Path | None = _DEFAULT_BASE) -> Config:
    """Load a YAML config, merging it onto the base config when present.

    Parameters
    ----------
    path:
        Path to the experiment config.
    base_path:
        Path to a base config to inherit from. If it does not exist, or if
        ``path`` *is* the base config, no merge is performed.
    """
    with open(path, encoding="utf-8") as handle:
        override = yaml.safe_load(handle) or {}

    resolved: Dict[str, Any] = override
    if base_path is not None:
        base_p, this_p = Path(base_path), Path(path)
        if base_p.exists() and base_p.resolve() != this_p.resolve():
            with open(base_p, encoding="utf-8") as handle:
                base = yaml.safe_load(handle) or {}
            resolved = _deep_merge(base, override)

    return Config(resolved)
