"""Parse string-typed experiment metadata into numeric features."""
from __future__ import annotations

import re

import pandas as pd


def parse_temperature_celsius(val) -> float:
    """Parse ``temperature`` cell to Celsius float; NaN if unparseable."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    s = str(val).strip().replace("°", "")
    if not s:
        return float("nan")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return float("nan")
    return float(m.group(0))


def parse_ph(val) -> float:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    if isinstance(val, (int, float)) and not pd.isna(val):
        return float(val)
    s = str(val).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_shaking_rpm(val) -> float:
    """Extract RPM from ``shaking``; NaN for non-numeric tokens like ``orbital``."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("nan")
    s = str(val).strip().lower()
    if not s:
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*rpm", s)
    if m:
        return float(m.group(1))
    m2 = re.search(r"^(\d+(?:\.\d+)?)$", s)
    if m2:
        return float(m2.group(1))
    return float("nan")
