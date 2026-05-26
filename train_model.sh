#!/bin/bash

set -e

# ---------- Colors ----------
RESET="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"

RED="\033[38;5;196m"
GREEN="\033[38;5;46m"
YELLOW="\033[38;5;226m"
BLUE="\033[38;5;39m"
PURPLE="\033[38;5;141m"
GRAY="\033[38;5;245m"

# ---------- Icons ----------
INFO_ICON="󰋽"
SUCCESS_ICON="󰄬"
ERROR_ICON="󰅚"
WARN_ICON="󰀦"

# ---------- Logging ----------
info() {
    echo -e "${BLUE}${BOLD}${INFO_ICON} INFO${RESET} ${DIM}$1${RESET}"
}

success() {
    echo -e "${GREEN}${BOLD}${SUCCESS_ICON} SUCCESS${RESET} ${DIM}$1${RESET}"
}

warn() {
    echo -e "${YELLOW}${BOLD}${WARN_ICON} WARN${RESET} ${DIM}$1${RESET}"
}

error() {
    echo -e "${RED}${BOLD}${ERROR_ICON} ERROR${RESET} ${DIM}$1${RESET}"
}

# Help check
if [[ -z "$1" || -z "$2" ]]; then
  info "MODEL_NAME and DATA_FOLDER are required arguments"
  info "Usage: ./train_model.sh <MODEL_NAME> <DATA_FOLDER>"
  info "Example: ./train_model.sh google/gemma-4-E2B-it ./terminal-data"
  info "Optional:"
  info "  <CLEAR_OLD_FILES=true|false>"
  info "Defaults To false"
  info ""
  info "Examples:"
  info "  ./train_model.sh google/gemma-4-E2B-it ./terminal-data"
  info "  ./train_model.sh google/gemma-4-E2B-it ./terminal-data false"
  info "  ./train_model.sh google/gemma-4-E2B-it ./terminal-data true"
  exit 1
fi

# Check for file desctruction
if [[ -z "$3" ]]; then
  CLEAR_FILES="false"
else
  case "$3" in
    [Tt][Rr][Uu][Ee])
      CLEAR_FILES="true"
      ;;
    [Ff][Aa][Ll][Ss][Ee])
      CLEAR_FILES="false"
      ;;
    *)
      CLEAR_FILES="false"
      ;;
  esac
fi

# Begin Training


MODEL_NAME="$1"
FINE_TUNE_FOLDER="$2"

# Create Venv if doesnt exist
# but Hide pip output but still show errors
python3 -m venv venv >/dev/null 2>&1
source ./venv/bin/activate >/dev/null 2>&1
python -m pip install -r requirements.txt \
  --disable-pip-version-check \
  -q


if ! python3 - <<EOF >/dev/null 2>&1
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODEL_NAME")
template = tok.chat_template or ""
exit(0 if template else 1)
EOF
then
    error "Model tokenizer does not define a chat template"
    exit 1
fi

success "Training $MODEL_NAME"


if [[ "$CLEAR_FILES" == "true" ]]; then
  info "Clearing Files"
  rm -rf ./adapters
  rm -rf ./output_model
fi


## Finding Max Sequence Length
info "Finding Max Sequence Length"

MAX_SEQ_LENGTH=$(python gen_max_sequence.py \
  --train-path="$FINE_TUNE_FOLDER" \
  --mlx-model="$MODEL_NAME" | tee /dev/tty | tail -n 1)

info "Max sequence length: $MAX_SEQ_LENGTH"

if ! [[ "$MAX_SEQ_LENGTH" =~ ^[0-9]+$ ]]; then
  error "Invalid max sequence length: $MAX_SEQ_LENGTH"
  exit 1
fi

ROUND_TO=128
MAX_SEQ_LENGTH=$(( ((MAX_SEQ_LENGTH + ROUND_TO - 1) / ROUND_TO) * ROUND_TO ))
info "Rounded max sequence length: $MAX_SEQ_LENGTH"

info "Running Training Safety Check With 50 Iterations"
mlx_lm lora \
  --model "$MODEL_NAME" \
  --train \
  --data "$FINE_TUNE_FOLDER" \
  --adapter-path ./adapters_smoke \
  --iters 50 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --grad-accumulation-steps 4 \
  --mask-prompt \
  --save-every 50

info "Running NaN Check On Smoke Test Adapters"
python checkNaNs.py --adapter-path=./adapters_smoke/adapters.safetensors
rm -rf ./adapters_smoke

info "Running Full Training With 500 Iterations"
mlx_lm lora \
  --model "$MODEL_NAME" \
  --train \
  --data "$FINE_TUNE_FOLDER" \
  --adapter-path ./adapters \
  --iters 500 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --grad-accumulation-steps 4 \
  --mask-prompt \
  --save-every 50

info "Running Final NaN Check"
python checkNaNs.py --adapter-path=./adapters/adapters.safetensors

success "Done Training"
info "Fusing Trained Model"

mlx_lm fuse \
  --model "$MODEL_NAME" \
  --adapter-path ./adapters \
  --save-path ./output_model

success "Done Fusing Model. Output saved to ./output_model"

deactivate
