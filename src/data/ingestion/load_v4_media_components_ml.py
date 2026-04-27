"""Load Media_Components_ML sheet from media_composition_v4.xlsx.

Stub — implement during Stage 1 condition encoding specification.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


WORKBOOK_V4 = Path("data/media_composition_v4.xlsx")
SHEET_NAME = "Media_Components_ML"


def load_media_components_ml(path: Path | None = None) -> pd.DataFrame:
    """Load the ML-ready components sheet.

    Returns a DataFrame with columns including at minimum:
      Media, Canonical_ID, Include_in_ml, Amount, Unit.
    Rows with Include_in_ml != True are retained but flagged.
    """
    path = path or WORKBOOK_V4
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    return df
