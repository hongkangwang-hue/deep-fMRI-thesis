import yaml
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load and return the project config. Resolves relative paths to absolute."""
    if path is None:
        path = _PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _resolve_paths(cfg, _PROJECT_ROOT)
    return cfg


def _resolve_paths(cfg: dict, root: Path) -> None:
    """Convert path strings in cfg['paths'] and cfg['datasets'] to absolute Paths."""
    for key in ("data_dir", "textgrid_dir", "respdict", "em_data_dir"):
        if key in cfg.get("datasets", {}):
            cfg["datasets"][key] = str(root / cfg["datasets"][key])
    for key in ("frozen_dir", "cache_dir", "results_dir", "figures_dir"):
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = str(root / cfg["paths"][key])
