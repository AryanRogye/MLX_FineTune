#!/bin/bash

set -e

# Help check
if [[ -z "$1" || -z "$2" ]]; then
  echo "Usage: ./train_model.sh <MODEL_NAME> <DATA_FOLDER>"
  echo "Example: ./train_model.sh google/gemma-4-E2B-it ./terminal-data"
  echo "Optional:"
  echo "  <CLEAR_OLD_FILES=true|false>"
  echo "Defaults To false"
  echo ""
  echo "Examples:"
  echo "  ./train_model.sh google/gemma-4-E2B-it ./terminal-data"
  echo "  ./train_model.sh google/gemma-4-E2B-it ./terminal-data false"
  echo "  ./train_model.sh google/gemma-4-E2B-it ./terminal-data true"
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
python3 -m venv venv
source ./venv/bin/activate
python -m pip install -r requirements.txt

python3 - <<EOF
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODEL_NAME")
template = tok.chat_template or ""
if not template:
    print("WARNING: Model tokenizer does not define a chat template")
    print("mlx_lm may require a model-specific template for messages-format data")
    exit(1)
EOF

echo "Training $MODEL_NAME"

if [[ "$CLEAR_FILES" == "true" ]]; then
  echo "Clearing Files"
  rm -rf ./adapters
  rm -rf ./output_model
fi

## Finding Max Sequence Length
echo "Finding Max Sequence Length"

MAX_SEQ_LENGTH=$(python gen_max_sequence.py \
  --train-path="$FINE_TUNE_FOLDER" \
  --mlx-model="$MODEL_NAME" | tee /dev/tty | tail -n 1)

echo "Max sequence length: $MAX_SEQ_LENGTH"

if ! [[ "$MAX_SEQ_LENGTH" =~ ^[0-9]+$ ]]; then
  echo "Invalid max sequence length: $MAX_SEQ_LENGTH"
  exit 1
fi

ROUND_TO=128
MAX_SEQ_LENGTH=$(( ((MAX_SEQ_LENGTH + ROUND_TO - 1) / ROUND_TO) * ROUND_TO ))
echo "Rounded max sequence length: $MAX_SEQ_LENGTH"

echo "Running Training Safety Check With 50 Iterations"
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

python checkNaNs.py --adapter-path=./adapters_smoke/adapters.safetensors
rm -rf ./adapters_smoke

echo "Running Full Training With 500 Iterations"
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

python checkNaNs.py --adapter-path=./adapters/adapters.safetensors

echo "Done Training"
echo "Fusing Trained Model"

mlx_lm fuse \
  --model "$MODEL_NAME" \
  --adapter-path ./adapters \
  --save-path ./output_model

deactivate
