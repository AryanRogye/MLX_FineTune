import os
from transformers import AutoTokenizer
import json
import argparse

# Create Arguements
parser = argparse.ArgumentParser()
parser.add_argument("--train-path", type=str, help="Path to the jsonl files directory")
parser.add_argument("--mlx-model", type=str, help="Hugging Face MLX Model")
args = parser.parse_args()

if args.train_path is None:
    print("Please provide the path to the jsonl files directory using --train-path")
    exit(1)

if args.mlx_model is None:
    print("Please provide a MLX model using --mlx-model")
    exit(1)

path = args.train_path
model = args.mlx_model

if not os.path.exists(path):
    print(f"Directory not found at {path}")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(model)

# First Check if the files exist
paths: list[str] = []
for filename in ["train.jsonl", "test.jsonl", "valid.jsonl"]:
    file_path = os.path.join(path, filename)
    paths.append(file_path)
    if not os.path.exists(file_path):
        print(f"File not found at {file_path}")
        exit(1)


maxes: list[int] = []
print("===============================")
for path in paths:
    lengths = []

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
            tokens = tokenizer(text)["input_ids"]
            lengths.append(len(tokens))

    max_length = max(lengths)
    maxes.append(max_length)

    print("Statistics for file: ", path)
    print(f"max: {max_length}")
    print(f"mean: {int(sum(lengths)/len(lengths))}")
    print(f"95th percentile: {sorted(lengths)[int(len(lengths)*0.95)]}")
    print("=================================")

print(max(maxes))
