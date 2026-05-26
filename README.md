# Scratch Repo

this repo is me learning how to fine tune a MLX Model

the script `train_model.sh` requires 2 arguments

## Arguments

### 1. `MODEL_NAME`
A HuggingFace model ID. Must have a defined chat template in its tokenizer.
- mlx-community/Qwen2.5-1.5B-Instruct-4bit
- google/gemma-4-E2B-it

### 2. `DATA_FOLDER`
Path to a folder containing exactly three `.jsonl` split files:

| File | Purpose |
|------|---------|
| `train.jsonl` | Examples the model trains on |
| `valid.jsonl` | Used to measure loss during training |
| `test.jsonl` | Final held-out evaluation |

```bash
⮕ ls playground/messages/outputs/2026-05-26_13-46-04
  test.jsonl   train.jsonl   valid.jsonl
```
