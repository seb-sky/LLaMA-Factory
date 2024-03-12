#!/usr/bin/env just --justfile

set dotenv-load

default:
  just --list

examples_qlora_single_gpu_bitsandbytes:
  source .venv/bin/activate && \
  cd examples/qlora_single_gpu && \
  source ./bitsandbytes.sh

seb:
  source .venv/bin/activate && \
  source ./seb.sh

tensorboard:
  source .venv/bin/activate && \
  tensorboard --logdir tmp/out/tensorboard --bind_all --port 6008