import argparse
import os

import numpy as np
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peek at tensors inside a .safetensors file."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the .safetensors file.",
    )
    parser.add_argument(
        "--tensor",
        help="Optional tensor name to print.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="How many rows to print for a tensor slice.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="How many columns to print for a tensor slice.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="How many tensor names to list when --tensor is not provided.",
    )
    parser.add_argument(
        "--lora-prefix",
        help=(
            "Optional LoRA tensor prefix, like "
            "model.layers.12.self_attn.q_proj. "
            "The script will load <prefix>.lora_a and <prefix>.lora_b "
            "and print a slice of A @ B."
        ),
    )
    return parser.parse_args()


def print_tensor_slice(name: str, arr: np.ndarray, rows: int, cols: int) -> None:
    print(f"{name}")
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")

    if arr.ndim == 0:
        print(arr.item())
        return

    if arr.ndim == 1:
        print(np.array2string(arr[:cols], precision=6, suppress_small=False))
        return

    print(np.array2string(arr[:rows, :cols], precision=6, suppress_small=False))


def list_tensors(path: str, limit: int) -> None:
    with safe_open(path, framework="numpy") as tensors:
        keys = list(tensors.keys())
        print(f"file: {path}")
        print(f"tensor count: {len(keys)}")
        print()

        for name in keys[:limit]:
            arr = tensors.get_tensor(name)
            print(f"{name:80s} shape={arr.shape} dtype={arr.dtype}")

        if len(keys) > limit:
            print()
            print(f"... {len(keys) - limit} more tensors")


def print_lora_delta(path: str, prefix: str, rows: int, cols: int) -> None:
    a_name = prefix + ".lora_a"
    b_name = prefix + ".lora_b"

    with safe_open(path, framework="numpy") as tensors:
        keys = set(tensors.keys())
        missing = [name for name in (a_name, b_name) if name not in keys]
        if missing:
            print("Missing LoRA tensor(s):")
            for name in missing:
                print(name)
            raise SystemExit(1)

        a = tensors.get_tensor(a_name)
        b = tensors.get_tensor(b_name)
        delta = a[:rows, :] @ b[:, :cols]

    print(f"LoRA prefix: {prefix}")
    print(f"A: {a_name} shape={a.shape} dtype={a.dtype}")
    print(f"B: {b_name} shape={b.shape} dtype={b.dtype}")
    print(f"A @ B top-left {rows}x{cols}:")
    print(np.array2string(delta, precision=7, suppress_small=False))


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"Safetensors file not found: {args.path}")
        raise SystemExit(1)

    if args.rows <= 0 or args.cols <= 0:
        print("--rows and --cols must be positive integers.")
        raise SystemExit(1)

    if args.lora_prefix:
        print_lora_delta(args.path, args.lora_prefix, args.rows, args.cols)
        return

    if args.tensor:
        with safe_open(args.path, framework="numpy") as tensors:
            keys = set(tensors.keys())
            if args.tensor not in keys:
                print(f"Tensor not found: {args.tensor}")
                raise SystemExit(1)

            arr = tensors.get_tensor(args.tensor)
            print_tensor_slice(args.tensor, arr, args.rows, args.cols)
        return

    list_tensors(args.path, args.limit)


if __name__ == "__main__":
    main()
