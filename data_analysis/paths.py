"""Repo-root-relative paths for Phase 0 analysis."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_paths import EMBEDDING_LAYER8_DIR, FEBA_DB, MEDIA_XLSX, REPO_ROOT  # noqa: F401, E402

# Alias used by Phase 0 scripts.
EMBEDDINGS_DIR = EMBEDDING_LAYER8_DIR

FIGURES_PHASE0 = REPO_ROOT / "figures" / "phase0"
OUTPUT_DIR = REPO_ROOT / "data_analysis" / "outputs"
