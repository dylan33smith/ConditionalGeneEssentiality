"""Encode proteins with ESM-C 600M for T5-C bypass experiment.

Reads protein sequences from data/raw/aaseqs (single FASTA file, header
format `>orgId:locusId`), runs ESM-C 600M with token-budgeted batching,
mean-pools per protein to get a 1152-dim vector, and writes per-organism
`.pt` files matching the format used by the existing ProtLM_embeddings_layer8/
directory:

    {orgId}_esmc.pt -> {
        "embeddings": Tensor[n_proteins, 1152] (bfloat16),
        "group_labels": list of "orgId:locusId" strings,
    }

Output dir: data/processed/ESMC_embeddings/
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import torch
from esm.models.esmc import ESMC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_fasta(fasta_path: Path, max_len: int = 4096) -> dict[str, list[tuple[str, str]]]:
    """Parse aaseqs FASTA into {orgId: [(gene_key, sequence), ...]}."""
    by_org: dict[str, list[tuple[str, str]]] = defaultdict(list)
    current_id: str | None = None
    current_seq: list[str] = []

    def _flush():
        if current_id is None:
            return
        seq = "".join(current_seq)[:max_len]
        org = current_id.split(":", 1)[0]
        by_org[org].append((current_id, seq))

    with fasta_path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                _flush()
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        _flush()
    return dict(by_org)


def average_representation(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Mean-pool hidden states, excluding pad positions."""
    mask = (input_ids != pad_token_id).unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


@torch.no_grad()
def encode_organism(
    model: ESMC,
    proteins: list[tuple[str, str]],
    device: torch.device,
    max_tokens_per_batch: int = 16000,
) -> tuple[torch.Tensor, list[str]]:
    """Encode all proteins for one organism with token-budgeted batching."""
    # Sort by length so each batch has roughly homogeneous lengths
    indexed = list(enumerate(proteins))
    indexed.sort(key=lambda x: len(x[1][1]))

    embeddings_out: dict[int, torch.Tensor] = {}
    pad_id = model.tokenizer.pad_token_id

    current_batch: list[tuple[int, str, str]] = []
    current_tokens = 0

    def _flush():
        nonlocal current_batch, current_tokens
        if not current_batch:
            return
        seqs = [s for _, _, s in current_batch]
        input_ids = model._tokenize(seqs).long().to(device)
        output = model(input_ids)
        emb = average_representation(output.embeddings, input_ids, pad_id)
        emb = emb.to(torch.bfloat16).cpu()
        for (orig_idx, _, _), e in zip(current_batch, emb, strict=True):
            embeddings_out[orig_idx] = e
        current_batch = []
        current_tokens = 0

    for orig_idx, (gene_key, seq) in indexed:
        seq_len = len(seq) + 2
        if current_batch and current_tokens + seq_len > max_tokens_per_batch:
            _flush()
        current_batch.append((orig_idx, gene_key, seq))
        current_tokens += seq_len
    _flush()

    n = len(proteins)
    embed_dim = next(iter(embeddings_out.values())).shape[0]
    out = torch.zeros(n, embed_dim, dtype=torch.bfloat16)
    for orig_idx, emb in embeddings_out.items():
        out[orig_idx] = emb
    labels = [gk for gk, _ in proteins]
    return out, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, default=Path("data/raw/aaseqs"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/ESMC_embeddings"),
    )
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--organisms", nargs="*", default=None,
        help="Restrict to these organism IDs (default: all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model: %s on %s", args.model, args.device)
    device = torch.device(args.device)
    model = ESMC.from_pretrained(args.model).eval().to(device, dtype=torch.bfloat16)

    log.info("Parsing FASTA: %s", args.fasta)
    by_org = parse_fasta(args.fasta)
    log.info("Found %d organisms, %d total proteins",
             len(by_org), sum(len(v) for v in by_org.values()))

    if args.organisms:
        by_org = {k: v for k, v in by_org.items() if k in set(args.organisms)}
        log.info("Restricted to %d organisms", len(by_org))

    sorted_orgs = sorted(by_org.keys())
    for i, org in enumerate(sorted_orgs):
        out_path = args.output_dir / f"{org}_esmc.pt"
        if out_path.exists():
            log.info("[%d/%d] skip %s (exists)", i + 1, len(sorted_orgs), org)
            continue
        proteins = by_org[org]
        log.info("[%d/%d] encoding %s (%d proteins)",
                 i + 1, len(sorted_orgs), org, len(proteins))
        embeddings, labels = encode_organism(model, proteins, device)
        torch.save(
            {"embeddings": embeddings, "group_labels": labels},
            out_path,
        )
        log.info("    saved %s (shape=%s)", out_path, tuple(embeddings.shape))

    log.info("ESM-C encoding complete.")


if __name__ == "__main__":
    main()
