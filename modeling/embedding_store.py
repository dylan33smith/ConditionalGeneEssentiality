"""Frozen ProteomeLM tensors: per-org lookup gene_key -> row index + weight matrix."""

from __future__ import annotations

from pathlib import Path

import torch


class OrgEmbeddingTable:
    __slots__ = ("embeddings", "gene_to_row")

    def __init__(self, embeddings: torch.Tensor, gene_to_row: dict[str, int]) -> None:
        self.embeddings = embeddings
        self.gene_to_row = gene_to_row


class EmbeddingStore:
    """Loads *_proteomelm.pt per organism; CPU tensors by default."""

    def __init__(self, embed_dir: Path, org_ids: set[str], device: torch.device) -> None:
        self.embed_dir = Path(embed_dir)
        self.device = device
        self._by_org: dict[str, OrgEmbeddingTable] = {}
        for org in sorted(org_ids):
            path = self.embed_dir / f"{org}_proteomelm.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing embedding file for org {org}: {path}")
            blob = torch.load(path, map_location="cpu", weights_only=False)
            emb = blob["embeddings"]
            labels = blob["group_labels"]
            if not isinstance(emb, torch.Tensor):
                emb = torch.as_tensor(emb)
            emb = emb.float().contiguous()
            gmap = {str(g): i for i, g in enumerate(labels)}
            self._by_org[org] = OrgEmbeddingTable(emb, gmap)

    @property
    def gene_embedding_dim(self) -> int:
        t0 = next(iter(self._by_org.values()))
        return int(t0.embeddings.shape[1])

    def vectors_for_rows(
        self,
        org_ids: list[str],
        gene_keys: list[str],
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather D-vectors for each (org, gene_key); writes into out[B, D] if provided."""
        b = len(org_ids)
        if b == 0:
            if out is not None:
                return out
            raise ValueError("empty batch")
        d = self._by_org[org_ids[0]].embeddings.shape[1]
        if out is None:
            out = torch.empty(b, d, dtype=torch.float32, device=self.device)
        for i, (o, gk) in enumerate(zip(org_ids, gene_keys, strict=True)):
            tab = self._by_org[o]
            j = tab.gene_to_row.get(gk)
            if j is None:
                raise KeyError(f"No embedding for gene_key={gk!r} org={o!r}")
            out[i].copy_(tab.embeddings[j], non_blocking=True)
        return out

    def has_embedding(self, org: str, gene_key: str) -> bool:
        tab = self._by_org[org]
        return gene_key in tab.gene_to_row
