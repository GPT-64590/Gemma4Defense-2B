#!/usr/bin/env bash
# Fine-tune google/gemma-4-E2B-it via LoRA on the assembled training corpus.
#
# Reproduces the released Gemma4Defense-2B checkpoint (within multi-trial
# noise) using the v3.4 single-run-equivalent recipe documented in
# docs/RESEARCH_NOTES.md.
#
# Prerequisites:
#   - bash build_corpus.sh has been run (creates data/train/combined_train.jsonl)
#   - 1× GPU with ≥ 24 GB VRAM (training)
#   - Python 3.11+, torch>=2.6, transformers>=4.51, peft, trl==0.29.1, accelerate
#   - HF auth: `huggingface-cli login` or HF_TOKEN env var (gated Gemma access)
#
# Hyperparameters (LoRA r=64, alpha=64, dropout=0.05, lr=5e-5, 10 epochs, bf16):
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p output/adapter

if [ ! -f data/train/combined_train.jsonl ]; then
  echo "Running build_corpus.sh first..."
  bash build_corpus.sh
fi

python src/train.py \
  --data data/train/combined_train.jsonl \
  --output-dir output/adapter/gemma4defense-2b \
  --base-model google/gemma-4-E2B-it \
  --lora-r 64 --lora-alpha 64 --lora-dropout 0.05 \
  --max-seq-length 4096 \
  --per-device-batch-size 2 --grad-accum 8 \
  --num-epochs 10 \
  --lr 5e-5 --warmup-ratio 0.05 --weight-decay 0.01 \
  --logging-steps 10 --save-steps 200 --seed 42 \
  --gradient-checkpointing

echo
echo "Adapter saved to output/adapter/gemma4defense-2b/"
echo "Next step: merge adapter into base for inference, e.g.:"
echo
echo "  python -c \""
echo "import torch"
echo "from transformers import AutoModelForCausalLM, AutoTokenizer"
echo "from peft import PeftModel"
echo "m = AutoModelForCausalLM.from_pretrained('google/gemma-4-E2B-it', dtype=torch.bfloat16)"
echo "m = PeftModel.from_pretrained(m, 'output/adapter/gemma4defense-2b').merge_and_unload()"
echo "m.save_pretrained('output/merged/gemma4defense-2b')"
echo "AutoTokenizer.from_pretrained('google/gemma-4-E2B-it').save_pretrained('output/merged/gemma4defense-2b')"
echo "  \""
