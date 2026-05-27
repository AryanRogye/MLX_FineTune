import argparse
from typing import Union, Tuple, Dict, Any

import mlx.nn as nn # type: ignore
from mlx_lm import load, convert # type: ignore
from mlx_lm.tokenizer_utils import TokenizerWrapper


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Add Arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for quantization"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./mlx_model"
    )

    return parser.parse_args()

args = get_args()

# Ask if we want to convert the model to 4-bit quantization
convert_to_4bit = input("Do you want to convert the model to 4-bit quantization? (yes/no): ").strip().lower()
if convert_to_4bit == "yes":
    print("Converting model to 4-bit quantization...")
    convert(
        args.model,
        mlx_path=str(args.out),
        quantize=True,
        q_bits=4
    )
