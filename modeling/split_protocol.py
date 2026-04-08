"""Load organism split protocol JSON (M3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitProtocol:
    protocol_id: str
    train_org_ids: frozenset[str]
    val_org_ids: frozenset[str]
    test_org_ids: frozenset[str]
    canonical_fitness_parquet: str
    canonical_fitness_sha256_expected: str | None
    source_path: Path

    def all_org_ids(self) -> frozenset[str]:
        return self.train_org_ids | self.val_org_ids | self.test_org_ids


def load_split_protocol(path: Path) -> SplitProtocol:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SplitProtocol(
        protocol_id=str(data["protocol_id"]),
        train_org_ids=frozenset(data["train_org_ids"]),
        val_org_ids=frozenset(data["val_org_ids"]),
        test_org_ids=frozenset(data["test_org_ids"]),
        canonical_fitness_parquet=str(data.get("canonical_fitness_parquet", "")),
        canonical_fitness_sha256_expected=data.get("canonical_fitness_sha256_expected"),
        source_path=path.resolve(),
    )
