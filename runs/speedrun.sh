#!/bin/bash
# flaxchat speedrun: Full pipeline on TPU pod
# Tokenizer -> Pretrain -> SFT -> Chat UI
#
# Usage:
#   bash runs/speedrun.sh
#
# Requirements:
#   - TPU VM with JAX installed
#   - pixi installed (curl -fsSL https://pixi.sh/install.sh | bash)

set -e

echo "============================================"
echo "  flaxchat speedrun"
echo "  Full pipeline: tokenizer -> pretrain -> SFT -> chat"
echo "============================================"

# Install dependencies
echo "[1/5] Installing dependencies..."
pixi install

# Download data and train tokenizer
echo "[2/5] Training tokenizer..."
pixi run -- python -m scripts.tok_train --vocab-size=32768 --num-shards=8

# Pretrain base model
echo "[3/5] Pretraining base model (d24)..."
pixi run -- python -m scripts.pretrain \
    --depth=24 \
    --run=speedrun-d24 \
    --eval-every=250 \
    --sample-every=2000 \
    --save-every=-1

# Supervised fine-tuning
echo "[4/5] Supervised fine-tuning..."
pixi run -- python -m scripts.sft \
    --base-model=d24 \
    --dataset=smoltalk \
    --num-iterations=500 \
    --run=speedrun-sft

# Launch web UI
echo "[5/5] Launching chat web UI..."
echo "Open http://localhost:8000 in your browser"
pixi run -- python -m scripts.chat_web --model=d24 --port=8000
