#!/bin/bash
set -e

########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"


if [ $# -lt 2 ]; then
  echo "Usage: GPU=0,1,2,... bash $0 SAVE_DIR CKPT"
  exit 1
fi

SAVE_DIR=$1
CKPT=$2

# Parse GPU ids
IFS=',' read -r -a GPUS <<< "$GPU"
NUM_GPUS=${#GPUS[@]}

# Define your parameter sets (8 total)
PARAMS=(
  "--sample_opt.manual_noise_h_std 0.0 --sample_opt.manual_noise_v_std 0.0 --sample_opt.manual_noise_x_std 0.0"
  "--sample_opt.manual_noise_h_std 0.1 --sample_opt.manual_noise_v_std 0.0 --sample_opt.manual_noise_x_std 0.0"
  "--sample_opt.manual_noise_h_std 0.0 --sample_opt.manual_noise_v_std 0.1 --sample_opt.manual_noise_x_std 0.0"
  "--sample_opt.manual_noise_h_std 0.0 --sample_opt.manual_noise_v_std 0.0 --sample_opt.manual_noise_x_std 0.1"
  "--sample_opt.manual_noise_h_std 0.3 --sample_opt.manual_noise_v_std 0.0 --sample_opt.manual_noise_x_std 0.0"
  "--sample_opt.manual_noise_h_std 0.0 --sample_opt.manual_noise_v_std 0.3 --sample_opt.manual_noise_x_std 0.0"
  "--sample_opt.manual_noise_h_std 0.0 --sample_opt.manual_noise_v_std 0.0 --sample_opt.manual_noise_x_std 0.3"
  "--sample_opt.manual_noise_h_std 0.1 --sample_opt.manual_noise_v_std 0.1 --sample_opt.manual_noise_x_std 0.1"
)
NAMES=(
  "no_noise"
  "h01"
  "v01"
  "x01"
  "h03"
  "v03"
  "x03"
  "h01_v01_x01"
)

# -------------------
# Launch generation
# -------------------
for i in "${!PARAMS[@]}"; do
  GPU_ID=${GPUS[$((i % NUM_GPUS))]}   # cycle through GPUs
  SAVE_SUBDIR="$SAVE_DIR/${NAMES[$i]}"
  mkdir -p "$SAVE_SUBDIR"

  echo "Launching generation $i on GPU $GPU_ID -> $SAVE_SUBDIR, with logs under $SAVE_SUBDIR/generate.log"
  (
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate.py \
      --config configs/test/test_vae.yaml \
      --ckpt "$CKPT" \
      --save_dir "$SAVE_SUBDIR" \
      ${PARAMS[$i]} >"$SAVE_SUBDIR/generate.log" 2>&1
  ) &
done

wait
echo "All generation jobs finished."

# -------------------
# Launch evaluation
# -------------------
for i in "${!PARAMS[@]}"; do
  # GPU_ID=${GPUS[$((i % NUM_GPUS))]}   # cycle through GPUs again
  SAVE_SUBDIR="$SAVE_DIR/${NAMES[$i]}"

  echo "Launching evaluation $i on $SAVE_SUBDIR, with logs under ${SAVE_SUBDIR}/eval.log"
  (
    CUDA_VISIBLE_DEVICES=$GPU_ID python eval.py \
      --root_dir "$SAVE_SUBDIR" \
      --eval_config ./configs/test/eval_metrics_vae.yaml \
      --num_workers 12 >"$SAVE_SUBDIR/eval.log" 2>&1
  ) &
done

wait
echo "All evaluations finished."
