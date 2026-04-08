"""Paths for split artifacts (M3)."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_paths import CANONICAL_FITNESS_LONG, CANONICAL_MANIFEST, REPO_ROOT, SPLITS_ROOT  # noqa: F401, E402
