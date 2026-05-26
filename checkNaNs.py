import os
import sys
import argparse
import mlx.core as mlx


parser = argparse.ArgumentParser()
parser.add_argument("--adapter-path", type=str, help="Path to the adapter file")
args = parser.parse_args()

if args.adapter_path is None:
    print("Please provide the path to the adapter file using --adapter-path")
    exit(1)

path = args.adapter_path

if not os.path.exists(path):
    print(f"Adapter file not found at {path}")
    exit(1)

weights = mlx.load(path)


contains_nan = False
nan_count = 0

for name, tensor in weights.items():
    has_nan = mlx.any(mlx.isnan(tensor)).item()
    if has_nan:
        nan_count += 1
        contains_nan = True

if not contains_nan:
    print("No NaN values found in any tensor.")
    sys.exit(0)
else:
    print(f"NaN values were found in {nan_count} tensors. Please check the tensors listed above for more details.")
    sys.exit(1)