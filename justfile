#!/usr/bin/env just --justfile

set dotenv-load

default:
  just --list

examples_qlora_single_gpu_bitsandbytes:
  source .venv/bin/activate && \
  cd examples/qlora_single_gpu && \
  source ./bitsandbytes.sh

me:
  source .venv/bin/activate && \
  python src/train_bash.py \
  --model_name_or_path google/gemma-7b-it \
  --quantization_bit 4 \
  --flash_attn \
  --use_unsloth \
  --export_dir tmp/out/export \
  --template gemma \
  --dataset ultra_chat \
  --dataset_dir data \
  --overwrite_cache \
  --max_samples 100 \
  --val_size 0.1 \
  --output_dir tmp/out/output \
  --do_train \
  --do_eval \
  --evaluation_strategy epoch \
  --logging_dir tmp/out/logging \
  --bf16 \
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj


