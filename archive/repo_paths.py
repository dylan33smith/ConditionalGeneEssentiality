"""Single source of truth for repo-root-relative paths.

All per-package ``paths.py`` shims import from here.  Application code should
import from the package-local ``paths`` module (e.g. ``from paths import ...``
when running a script inside ``modeling/``), not directly from this file.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_RAW = REPO_ROOT / "data" / "raw"
FEBA_DB = DATA_RAW / "feba.db"
MEDIA_XLSX = REPO_ROOT / "data" / "media_composition.xlsx"       # v1 (original, 45 media)
MEDIA_XLSX_V2 = REPO_ROOT / "data" / "media_composition_v2.xlsx"  # v2 (120 media, physical components)
MEDIA_XLSX_V3 = REPO_ROOT / "data" / "media_composition_v3.xlsx"  # v3 (121 media, adds LB (Miller) in-silico)
MEDIA_XLSX_V4 = REPO_ROOT / "data" / "media_composition_v4.xlsx"  # v4 (121 media, 6 sheets incl. Media_Components_ML canonical chemistry)
MEDIA_XLSX_V5 = REPO_ROOT / "data" / "media_composition_v5.xlsx"  # v5 (191 media; +70 media of the new 62-org release; CURRENT for the 62-org rebuild)

DERIVED_CANONICAL_V0 = REPO_ROOT / "data" / "derived" / "canonical" / "v0"
CANONICAL_FITNESS_LONG = DERIVED_CANONICAL_V0 / "fitness_experiment_long.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "docs" / "canonical_build_manifest_v0.json"

SPLITS_ROOT = REPO_ROOT / "splits"
EMBEDDING_LAYER8_DIR = REPO_ROOT / "data" / "processed" / "ProtLM_embeddings_layer8"
