"""Load frozen ProteomeLM gene embeddings.

Stub — implement during Stage 0.
"""
from __future__ import annotations
from pathlib import Path
import torch


EMBEDDING_DIR = Path("data/processed/ProtLM_embeddings_layer8")


def load_org_embeddings(org_id: str, embedding_dir: Path | None = None) -> dict:
    """Load {gene_key: tensor} map for one organism.

    Returns a dict mapping gene_key strings to 1-D float tensors.
    """
    embedding_dir = embedding_dir or EMBEDDING_DIR
    pt_path = embedding_dir / f"{org_id}_proteomelm.pt"
    bundle = torch.load(pt_path, map_location="cpu", weights_only=True)
    return bundle
