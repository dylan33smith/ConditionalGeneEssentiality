"""Repo-root-relative paths for data_processing."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_paths import DATA_RAW, DERIVED_CANONICAL_V0, FEBA_DB, MEDIA_XLSX, REPO_ROOT  # noqa: F401, E402
