"""CLI entrypoint for running experiments.

Stub — implement during Stage 0.
Usage: python -m src.cli.run_experiment --config configs/tier1/exp_1a_granularity.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiered experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    raise NotImplementedError(
        "CLI entrypoint stub — implement src/train/loop.py first."
    )


if __name__ == "__main__":
    main()
