"""Condition encoding column naming (must match ``build_condition_encoding_v0`` output)."""

from __future__ import annotations


def ce_cat_column(logical_name: str) -> str:
    return f"ce_cat_{logical_name}"


def ce_cont_column(logical_name: str) -> str:
    return f"ce_cont_{logical_name}"
