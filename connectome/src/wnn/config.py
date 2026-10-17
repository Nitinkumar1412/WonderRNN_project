"""Config: YAML + ${ENV:-default} expansion + `<root>` path placeholder +
`key=value` CLI overrides.

    cfg = load_config("configs/default.yaml", ["train.epochs=2"])
    cfg.train.epochs              # 2
    cfg.paths.work_dir            # absolute path

`paths.root` defaults to the project root two levels above this package
(`<root>/code/connectome/src/wnn`), so the same relative layout works on a
laptop and on the cluster:

    <root>/wnn_connectome_starterpack/datasets   competition parquet files
    <root>/work                                  arrays, runs, teachers, submissions
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

_ENV = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")
CODE_DIR = Path(__file__).resolve().parents[2]          # .../code/connectome
DEFAULT_CONFIG = CODE_DIR / "configs" / "default.yaml"


class Cfg(dict):
    """Nested dict with attribute access: cfg.section.key."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key '{key}'") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Cfg({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _expand(x: Any) -> Any:
    if isinstance(x, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), x)
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x


def parse_scalar(s: str) -> Any:
    low = s.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s[:1] in "[{":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return s


def apply_override(cfg: Cfg, dotted: str) -> None:
    key, sep, val = dotted.partition("=")
    if not sep:
        raise ValueError(f"override must be key=value, got {dotted!r}")
    parts = key.strip().split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(f"unknown config section '{p}' in override {dotted!r}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key '{key}' (typo?)")
    node[parts[-1]] = _wrap(parse_scalar(val))


def _resolve_paths(cfg: Cfg) -> None:
    p = cfg.paths
    root = Path(p.root).expanduser() if p.get("root") else CODE_DIR.parents[1]
    root = root.resolve()
    p.root = str(root)
    for k, v in list(p.items()):
        if k != "root" and isinstance(v, str):
            p[k] = str(Path(v.replace("<root>", str(root))).expanduser())


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Cfg:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _wrap(_expand(raw))
    for o in overrides:
        apply_override(cfg, o)
    _resolve_paths(cfg)
    return cfg


def from_dict(d: dict) -> Cfg:
    return _wrap(d)


def to_json(cfg: Cfg) -> str:
    return json.dumps(cfg, indent=2, sort_keys=True, default=str)
