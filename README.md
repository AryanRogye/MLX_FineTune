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

---
---
---

# Notes
# Rank
how much unique information a matrix contains

matrix with
$$
\begin{bmatrix}
1 & 2 & 3\\ 
2 & 4 & 6\\
5 & 0 & 1\\
\end{bmatrix}
$$

rank is 2 even though there are 3 columns
because row 0 * 2 is just row 1

even for like unorodered
$$
\begin{bmatrix}
1 & 2 & 3\\
5 & 0 & 1\\
2 & 4 & 6\\
\end{bmatrix}
$$ 

rank is 2 row 0 * 2 = row 2

# LoRA (low rank adaptation)
lets say we have a matrix like this:

matrix with
$$
\begin{bmatrix}
1 & 2 & 3\\ 
2 & 4 & 6\\
5 & 0 & 1\\
\end{bmatrix}
$$

and we wanted to update the middle column

$$\begin{bmatrix}
1 & \mathbf{2} & 3 \\
2 & \mathbf{4} & 6 \\
5 & \mathbf{0} & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
0 & \mathbf{2} & 0 \\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 & \mathbf{4} & 3 \\
2 & \mathbf{8} & 6 \\
5 & \mathbf{0} & 1
\end{bmatrix}$$

this is kinda like updating params but
LoRA is saying that we compress down to a SMALLER space
do the transforms there and expand back up

this is sick and can inspect it
if we have a model that was trained/fine-tuned with LoRA

in my repo after fine tuning I have a adapters folder inside it a 
adapters.safetensors file:
```bash
⮕ ls adapters
 0000050_adapters.safetensors
 0000100_adapters.safetensors
 0000150_adapters.safetensors
 0000200_adapters.safetensors
 0000250_adapters.safetensors
 0000300_adapters.safetensors
 0000350_adapters.safetensors
 0000400_adapters.safetensors
 0000450_adapters.safetensors
 0000500_adapters.safetensors
 adapter_config.json
 adapters.safetensors
⮕
```
we can use our peeking script

```bash
⮕ source venv/bin/activate
⮕ python peek_safetensors.py \
>   --path ./adapters/adapters.safetensors
file: ./adapters/adapters.safetensors
tensor count: 224

model.layers.12.mlp.down_proj.lora_a                                             shape=(8960, 8) dtype=float32
model.layers.12.mlp.down_proj.lora_b                                             shape=(8, 1536) dtype=float32
model.layers.12.mlp.gate_proj.lora_a                                             shape=(1536, 8) dtype=float32
model.layers.12.mlp.gate_proj.lora_b                                             shape=(8, 8960) dtype=float32
model.layers.12.mlp.up_proj.lora_a                                               shape=(1536, 8) dtype=float32
model.layers.12.mlp.up_proj.lora_b                                               shape=(8, 8960) dtype=float32
model.layers.12.self_attn.k_proj.lora_a                                          shape=(1536, 8) dtype=float32
model.layers.12.self_attn.k_proj.lora_b                                          shape=(8, 256) dtype=float32
model.layers.12.self_attn.o_proj.lora_a                                          shape=(1536, 8) dtype=float32
model.layers.12.self_attn.o_proj.lora_b                                          shape=(8, 1536) dtype=float32
model.layers.12.self_attn.q_proj.lora_a                                          shape=(1536, 8) dtype=float32
model.layers.12.self_attn.q_proj.lora_b                                          shape=(8, 1536) dtype=float32
model.layers.12.self_attn.v_proj.lora_a                                          shape=(1536, 8) dtype=float32
model.layers.12.self_attn.v_proj.lora_b                                          shape=(8, 256) dtype=float32
model.layers.13.mlp.down_proj.lora_a                                             shape=(8960, 8) dtype=float32
model.layers.13.mlp.down_proj.lora_b                                             shape=(8, 1536) dtype=float32
model.layers.13.mlp.gate_proj.lora_a                                             shape=(1536, 8) dtype=float32
model.layers.13.mlp.gate_proj.lora_b                                             shape=(8, 8960) dtype=float32
model.layers.13.mlp.up_proj.lora_a                                               shape=(1536, 8) dtype=float32
model.layers.13.mlp.up_proj.lora_b                                               shape=(8, 8960) dtype=float32
model.layers.13.self_attn.k_proj.lora_a                                          shape=(1536, 8) dtype=float32
model.layers.13.self_attn.k_proj.lora_b                                          shape=(8, 256) dtype=float32
model.layers.13.self_attn.o_proj.lora_a                                          shape=(1536, 8) dtype=float32
model.layers.13.self_attn.o_proj.lora_b                                          shape=(8, 1536) dtype=float32
model.layers.13.self_attn.q_proj.lora_a                                          shape=(1536, 8) dtype=float32

... 199 more tensors
```

this is super interesting u can look closely at the first example
```bash
model.layers.12.mlp.down_proj.lora_a -> shape=(8960, 8) dtype=float32
model.layers.12.mlp.down_proj.lora_b -> shape=(8, 1536) dtype=float32
```
this tells us that our rank was 8
we compressed down to this, then did what we had to then decompressed back up

think about that like 

to update it wouldve been:
`8960 x 1536 = 13,762,560 numbers`
```
LoRA learns:

  8960 x 8  = 71,680 numbers
  8 x 1536  = 12,288 numbers
  total     = 83,968 numbers
```
thats fucking insane😂 thats lower than updating ALL 13mil

# Full Fine-Tuning

I put this after LoRA because it is basically the opposite idea.

Instead of compressing updates into a low-rank space,
we directly update ALL model parameters.

From the last example:

`8960 x 1536 = 13,762,560 trainable numbers`

instead of:

`8960 x 8   +   8 x 1536 = 83,968 trainable numbers`

# DoRA (Weight-Decomposed Low-Rank Adaptation)
