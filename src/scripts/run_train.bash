#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash src/scripts/run_train.bash <model_id> <dataset> [gpu]"
  echo
  echo "Examples:"
  echo "  bash src/scripts/run_train.bash sasrec Software 0"
  echo "  bash src/scripts/run_train.bash looprec_sasrec Office_Products 0"
  echo "  DATA_PATH_OVERRIDE=/path/to/processed_clean_v3_rearec bash src/scripts/run_train.bash looprec_lrurec Patio_Lawn_and_Garden 0"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_ID="$1"
DATASET_ID="$2"
GPU_ARG="${3:-0}"
CONFIG_FILE="$SCRIPT_DIR/config/$MODEL_ID.conf"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config not found: $CONFIG_FILE" >&2
  echo "Available configs:" >&2
  find "$SCRIPT_DIR/config" -maxdepth 1 -name '*.conf' \
    | sed "s|$SCRIPT_DIR/config/||; s|.conf$||" \
    | sort >&2
  exit 1
fi

source "$CONFIG_FILE"

DATA_PATH="${DATA_PATH_OVERRIDE:-${DATA_PATH:-$REPO_ROOT/datasets/processed_clean_v3_rearec}}"
DATA_PROFILE="${DATA_PROFILE_OVERRIDE:-${DATA_PROFILE:-amazon_abs_v3_rearec}}"
GPU="${GPU_OVERRIDE:-${GPU:-$GPU_ARG}}"

cmd=(
  python "$REPO_ROOT/src/main.py"
  --model_name "$MODEL_NAME"
  --dataset "$DATASET_ID"
  --path "$DATA_PATH"
  --data_profile "$DATA_PROFILE"
  --gpu "$GPU"
  --train "${TRAIN_OVERRIDE:-${TRAIN:-1}}"
  --load "${LOAD_OVERRIDE:-${LOAD:-0}}"
  --regenerate "${REGENERATE_OVERRIDE:-${REGENERATE:-0}}"
  --epoch "${EPOCH_OVERRIDE:-${EPOCH:-100}}"
  --early_stop "${EARLY_STOP_OVERRIDE:-${EARLY_STOP:-10}}"
  --batch_size "${BATCH_SIZE_OVERRIDE:-${BATCH_SIZE:-256}}"
  --lr "${LR_OVERRIDE:-${LR:-0.001}}"
  --l2 "${L2_OVERRIDE:-${L2:-0}}"
  --dropout "${DROPOUT_OVERRIDE:-${DROPOUT:-0.5}}"
  --warmup_epochs "${WARMUP_EPOCHS_OVERRIDE:-${WARMUP_EPOCHS:-0}}"
  --lr_scheduler "${LR_SCHEDULER_OVERRIDE:-${LR_SCHEDULER:-constant}}"
  --max_norm "${MAX_NORM_OVERRIDE:-${MAX_NORM:-1.0}}"
)

add_arg() {
  local value="$1"
  local flag="$2"
  if [[ -n "$value" ]]; then
    cmd+=("$flag" "$value")
  fi
}

add_arg "${EVAL_BATCH_SIZE_OVERRIDE:-${EVAL_BATCH_SIZE:-}}" "--eval_batch_size"
add_arg "${NUM_WORKERS_OVERRIDE:-${NUM_WORKERS:-}}" "--num_workers"
add_arg "${PIN_MEMORY_OVERRIDE:-${PIN_MEMORY:-}}" "--pin_memory"
add_arg "${ATTENTION_DROPOUT_OVERRIDE:-${ATTENTION_DROPOUT:-}}" "--attention_dropout"
add_arg "${EMB_SIZE_OVERRIDE:-${EMB_SIZE:-}}" "--emb_size"
add_arg "${NUM_LAYERS_OVERRIDE:-${NUM_LAYERS:-}}" "--num_layers"
add_arg "${NUM_HEADS_OVERRIDE:-${NUM_HEADS:-}}" "--num_heads"
add_arg "${INNER_SIZE_OVERRIDE:-${INNER_SIZE:-}}" "--inner_size"
add_arg "${USE_FULL_SEQ_LOSS_OVERRIDE:-${USE_FULL_SEQ_LOSS:-}}" "--use_full_seq_loss"
add_arg "${MODEL_PATH_OVERRIDE:-${MODEL_PATH:-}}" "--load_model_path"
add_arg "${RANDOM_SEED_OVERRIDE:-${RANDOM_SEED:-}}" "--random_seed"

add_arg "${LOOPREC_BACKBONE_OVERRIDE:-${LOOPREC_BACKBONE:-}}" "--looprec_backbone"
add_arg "${NUM_RECURSION_OVERRIDE:-${NUM_RECURSION:-}}" "--num_recursion"
add_arg "${LOOPREC_ROUTER_FORMULATION_OVERRIDE:-${LOOPREC_ROUTER_FORMULATION:-}}" "--looprec_router_formulation"
add_arg "${LOOPREC_ROUTER_TYPE_OVERRIDE:-${LOOPREC_ROUTER_TYPE:-}}" "--looprec_router_type"
add_arg "${LOOPREC_ROUTER_FUNC_OVERRIDE:-${LOOPREC_ROUTER_FUNC:-}}" "--looprec_router_func"
add_arg "${LOOPREC_ROUTER_TEMP_OVERRIDE:-${LOOPREC_ROUTER_TEMP:-}}" "--looprec_router_temp"
add_arg "${LOOPREC_ROUTER_INPUT_SOURCE_OVERRIDE:-${LOOPREC_ROUTER_INPUT_SOURCE:-}}" "--looprec_router_input_source"
add_arg "${LOOPREC_LOCALDEPTH_PARAMETRIZATION_OVERRIDE:-${LOOPREC_LOCALDEPTH_PARAMETRIZATION:-}}" "--looprec_localdepth_parametrization"
add_arg "${LOOPREC_LOCALDEPTH_TRAINING_MODE_OVERRIDE:-${LOOPREC_LOCALDEPTH_TRAINING_MODE:-}}" "--looprec_localdepth_training_mode"
add_arg "${LOOPREC_BALANCING_OVERRIDE:-${LOOPREC_BALANCING:-}}" "--looprec_balancing"
add_arg "${LOOPREC_BALANCING_COEFF_OVERRIDE:-${LOOPREC_BALANCING_COEFF:-}}" "--looprec_balancing_coeff"
add_arg "${LOOPREC_USE_Z_LOSS_OVERRIDE:-${LOOPREC_USE_Z_LOSS:-}}" "--looprec_use_z_loss"
add_arg "${LOOPREC_Z_LOSS_COEFF_OVERRIDE:-${LOOPREC_Z_LOSS_COEFF:-}}" "--looprec_z_loss_coeff"
add_arg "${LOOPREC_WARMUP_EPOCHS_OVERRIDE:-${LOOPREC_WARMUP_EPOCHS:-}}" "--looprec_warmup_epochs"
add_arg "${LOOPREC_KV_CACHE_STRATEGY_OVERRIDE:-${LOOPREC_KV_CACHE_STRATEGY:-}}" "--looprec_kv_cache_strategy"
add_arg "${LOOPREC_GATING_OVERRIDE:-${LOOPREC_GATING:-}}" "--looprec_gating"
add_arg "${LOOPREC_EXITED_TOKEN_MEMORY_OVERRIDE:-${LOOPREC_EXITED_TOKEN_MEMORY:-}}" "--looprec_exited_token_memory"

add_arg "${BSAREC_ALPHA_OVERRIDE:-${BSAREC_ALPHA:-}}" "--bsarec_alpha"
add_arg "${BSAREC_C_OVERRIDE:-${BSAREC_C:-}}" "--bsarec_c"
add_arg "${NEXTITNET_DILATIONS_OVERRIDE:-${NEXTITNET_DILATIONS:-}}" "--nextitnet_dilations"
add_arg "${NEXTITNET_KERNEL_SIZE_OVERRIDE:-${NEXTITNET_KERNEL_SIZE:-}}" "--nextitnet_kernel_size"
add_arg "${NEXTITNET_BLOCK_NUM_OVERRIDE:-${NEXTITNET_BLOCK_NUM:-}}" "--nextitnet_block_num"
add_arg "${NEXTITNET_REG_WEIGHT_OVERRIDE:-${NEXTITNET_REG_WEIGHT:-}}" "--nextitnet_reg_weight"
add_arg "${LRUREC_R_MIN_OVERRIDE:-${LRUREC_R_MIN:-}}" "--lrurec_r_min"
add_arg "${LRUREC_R_MAX_OVERRIDE:-${LRUREC_R_MAX:-}}" "--lrurec_r_max"

if [[ -n "${CLIP_GRAD_OVERRIDE:-${CLIP_GRAD:-}}" ]]; then
  cmd+=(--clip_grad)
fi

echo "Running: ${cmd[*]}"
cd "$REPO_ROOT"
exec "${cmd[@]}"
