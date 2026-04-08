"""Repo paths for the training harness (plan §Harness)."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_paths import (  # noqa: F401, E402
    CANONICAL_FITNESS_LONG,
    EMBEDDING_LAYER8_DIR,
    REPO_ROOT,
    SPLITS_ROOT,
)

RUNS_ROOT = REPO_ROOT / "runs"


def resolve_parquet_path(repo_relative_or_abs: str) -> Path:
    p = Path(repo_relative_or_abs)
    return p if p.is_absolute() else REPO_ROOT / p
