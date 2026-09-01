"""Re-encode proteins with ProteomeLM-L, extracting multiple layers for T5-B.

Reads ESM-C mean-pooled embeddings from data/processed/ESMC_embeddings/
(produced by encode_esmc.py), passes them through ProteomeLM-L per
organism, and saves the hidden state from each requested layer to a
separate per-organism .pt file.

Layout per requested layer L:
    data/processed/PLM_embeddings_layerL/{orgId}_proteomelm.pt -> {
        "embeddings": Tensor[n_proteins, 1152] (bfloat16),
        "group_labels": list of "orgId:locusId",
    }

The format matches the existing data/processed/ProtLM_embeddings_layer8/
directory so the downstream training pipeline can swap one in for another
just by changing the `embedding_dir` arg.

ProteomeLM-L config: dim=1152, n_layers=18. Hidden states tuple from
output_hidden_states=True has length n_layers+1 (index 0 = input, index N
= layer-N output).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_proteomelm(checkpoint_dir: Path, device: torch.device):
    """Load ProteomeLM-L from a local snapshot directory.

    Imports modeling_proteomelm.py directly via importlib to avoid the
    package's __init__.py, which pulls in train.py + wandb.
    """
    import importlib.util

    modeling_path = (
        Path.home() / "projects" / "ProteomeLM" / "proteomelm" / "modeling_proteomelm.py"
    )
    spec = importlib.util.spec_from_file_location(
        "modeling_proteomelm", modeling_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ProteomeLMForMaskedLM = module.ProteomeLMForMaskedLM

    model = ProteomeLMForMaskedLM.from_pretrained(str(checkpoint_dir))
    model = model.eval().to(device, dtype=torch.bfloat16)
    return model


@torch.no_grad()
def encode_one_organism(
    model,
    esmc_embeddings: torch.Tensor,
    keep_layers: tuple[int, ...],
    device: torch.device,
    batch_size: int = 256,
) -> dict[int, torch.Tensor]:
    """Run ProteomeLM-L on per-protein ESM-C embeddings, batched by proteins.

    Each forward pass processes `batch_size` proteins together (each as a
    "proteome chunk" — mimics how ProteomeLM is typically applied at inference
    when proteomes exceed the max_position_embeddings of 512).

    Returns:
        {layer_idx: Tensor[n_proteins, 1152]} for each layer in keep_layers.
    """
    n = len(esmc_embeddings)
    per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in keep_layers}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = esmc_embeddings[start:end].to(device, dtype=torch.bfloat16)
        chunk = chunk.unsqueeze(0)  # (1, chunk_size, 1152)
        output = model.forward(inputs_embeds=chunk, output_hidden_states=True)
        hidden_states = output.hidden_states  # tuple of (1, chunk_size, 1152)
        for L in keep_layers:
            h = hidden_states[L].squeeze(0).to(torch.bfloat16).cpu()
            per_layer[L].append(h)

    return {L: torch.cat(per_layer[L], dim=0) for L in keep_layers}


def _unpack_esmc_bundle(bundle, org: str):
    """Accept either ESM-C on-disk layout and return (embeddings, group_labels).

    Two formats exist in this project:
      * the documented bundle: {"embeddings": Tensor[n, d], "group_labels": [...]}
      * a plain mapping produced by the 2026-07-06 regeneration: {locusId: Tensor[d]}

    The plain mapping is sorted by locusId so the row order is deterministic and
    reproducible across runs -- dict insertion order is not a stable contract.
    Labels are emitted as "orgId:locusId" to match the bundle convention.
    """
    if isinstance(bundle, dict) and "embeddings" in bundle and "group_labels" in bundle:
        return bundle["embeddings"], bundle["group_labels"]
    if isinstance(bundle, dict) and bundle and torch.is_tensor(next(iter(bundle.values()))):
        keys = sorted(bundle)
        emb = torch.stack([bundle[k].to(torch.float32) for k in keys])
        return emb, [f"{org}:{k}" for k in keys]
    raise ValueError(
        f"Unrecognised ESM-C bundle layout for {org}: "
        f"type={type(bundle).__name__}, "
        f"keys={list(bundle)[:5] if isinstance(bundle, dict) else 'n/a'}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--esmc-dir", type=Path,
        default=Path("data/processed/ESMC_embeddings"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(
            "/data/ds85/huggingface_cache/"
            "models--Bitbol-Lab--ProteomeLM-L/snapshots/"
            "0f834036ae8477bb15b019437aa6374c57d7aa49"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/processed"),
        help="Outputs will be written to {root}/PLM_embeddings_layer{L}/",
    )
    parser.add_argument(
        "--keep-layers", type=int, nargs="+",
        default=[0, 4, 8, 12, 18],
        help="Which ProteomeLM-L hidden states to save (0=input, 18=last)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--organisms", nargs="*", default=None,
        help="Restrict to these organism IDs (default: all)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    log.info("Loading ProteomeLM-L from %s on %s", args.checkpoint, device)
    model = load_proteomelm(args.checkpoint, device)
    log.info("Model loaded. Internal dim=%d, num_layers=%d",
             model.config.dim, model.config.n_layers)

    layer_dirs = {}
    for L in args.keep_layers:
        out_dir = args.output_root / f"PLM_embeddings_layer{L}"
        out_dir.mkdir(parents=True, exist_ok=True)
        layer_dirs[L] = out_dir

    esmc_files = sorted(args.esmc_dir.glob("*_esmc.pt"))
    log.info("Found %d ESM-C embedding files in %s", len(esmc_files), args.esmc_dir)

    if args.organisms:
        keep = set(args.organisms)
        esmc_files = [
            p for p in esmc_files
            if p.name.replace("_esmc.pt", "") in keep
        ]
        log.info("Restricted to %d organisms", len(esmc_files))

    for i, esmc_path in enumerate(esmc_files):
        org = esmc_path.name.replace("_esmc.pt", "")

        # Skip if all output layers already exist
        all_exist = all(
            (layer_dirs[L] / f"{org}_proteomelm.pt").exists()
            for L in args.keep_layers
        )
        if all_exist:
            log.info("[%d/%d] skip %s (all layers exist)",
                     i + 1, len(esmc_files), org)
            continue

        log.info("[%d/%d] loading %s", i + 1, len(esmc_files), org)
        bundle = torch.load(esmc_path, map_location="cpu", weights_only=False)
        esmc_emb, labels = _unpack_esmc_bundle(bundle, org)
        log.info("    encoding %s (n_proteins=%d, batch_size=%d)",
                 org, len(esmc_emb), args.batch_size)

        per_layer = encode_one_organism(
            model, esmc_emb, tuple(args.keep_layers),
            device, batch_size=args.batch_size,
        )

        for L, emb in per_layer.items():
            out_path = layer_dirs[L] / f"{org}_proteomelm.pt"
            torch.save({"embeddings": emb, "group_labels": labels}, out_path)
        log.info("    saved %d layers", len(per_layer))

    log.info("ProteomeLM-L layer encoding complete.")


if __name__ == "__main__":
    main()
