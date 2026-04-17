#!/usr/bin/env bash
# Run organism-quality-tier ablation: full vs curated vs curated_strict.
#
# Each run uses the same hyperparameters and seed; only the training set
# changes (val=Btheta, test=DvH in all cases).
#
# Usage:
#   bash modeling/run_quality_ablation.sh              # defaults: 5 epochs, full data
#   EPOCHS=8 bash modeling/run_quality_ablation.sh     # override epochs
#   MAX_TRAIN=50000 MAX_VAL=20000 bash modeling/run_quality_ablation.sh  # capped smoke
#   LOG_EVERY_BATCHES=0 bash modeling/run_quality_ablation.sh            # silence mid-epoch logs
#
# Full-data epochs can run a long time with no output; by default we pass
# --log-every-n-batches 2000 so you see periodic progress. Override with 0 for quiet runs.
#
# Outputs land under runs/ with a shared run-tag "quality_ablation_v0".

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-0}"
BATCH="${BATCH:-512}"
HIDDEN="${HIDDEN:-512}"
LR="${LR:-1e-3}"
TAG="quality_ablation_v0"

# Optional row caps (0 = no cap)
MAX_TRAIN="${MAX_TRAIN:-0}"
MAX_VAL="${MAX_VAL:-0}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-2000}"

COMMON_FLAGS=(
    --epochs "$EPOCHS"
    --seed "$SEED"
    --batch-size "$BATCH"
    --hidden-dim "$HIDDEN"
    --lr "$LR"
    --run-tag "$TAG"
)

if [ "$MAX_TRAIN" -gt 0 ]; then
    COMMON_FLAGS+=(--max-train-rows "$MAX_TRAIN")
fi
if [ "$MAX_VAL" -gt 0 ]; then
    COMMON_FLAGS+=(--max-val-rows "$MAX_VAL" --skip-full-row-counts)
fi
if [ "$LOG_EVERY_BATCHES" -gt 0 ] 2>/dev/null; then
    COMMON_FLAGS+=(--log-every-n-batches "$LOG_EVERY_BATCHES")
fi

PROTOCOLS=(
    "splits/organism_single_holdout_largest_v0/protocol.json"
    "splits/organism_single_holdout_largest_curated_v0/protocol.json"
    "splits/organism_single_holdout_largest_curated_strict_v0/protocol.json"
)

LABELS=(full curated curated_strict)

echo "=== Organism quality-tier ablation ==="
echo "  epochs=$EPOCHS  seed=$SEED  batch=$BATCH  hidden=$HIDDEN  lr=$LR"
echo "  max_train=$MAX_TRAIN  max_val=$MAX_VAL  log_every_n_batches=${LOG_EVERY_BATCHES:-0}"
echo ""

for i in "${!PROTOCOLS[@]}"; do
    proto="${PROTOCOLS[$i]}"
    label="${LABELS[$i]}"
    echo "--- [$label] protocol=$proto ---"
    python modeling/train.py \
        --protocol "$proto" \
        "${COMMON_FLAGS[@]}" \
        2>&1
    echo ""
done

echo "=== All three runs complete. Compare runs/*quality_ablation_v0* ==="
