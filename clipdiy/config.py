import dataclasses
from typing import List, Optional

import yaml

from .segment import SegConfig


def _parse(value: str):
    return yaml.safe_load(value)


def load_seg_config(path: Optional[str] = None, overrides: Optional[List[str]] = None, **kwargs) -> SegConfig:
    """YAML file -> SegConfig, then `key=value` overrides (values parsed as YAML), then kwargs."""
    values = {}
    if path:
        with open(path) as f:
            values.update(yaml.safe_load(f) or {})
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        values[key.strip()] = _parse(val)
    values.update(kwargs)
    known = {f.name for f in dataclasses.fields(SegConfig)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    if "scales" in values:
        values["scales"] = tuple(values["scales"])
    return SegConfig(**values)


def save_seg_config(cfg: SegConfig, path: str):
    d = dataclasses.asdict(cfg)
    d["scales"] = list(d["scales"])
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
