#!/usr/bin/env just --justfile

set dotenv-load

default:
  just --list

examples_qlora_single_gpu_bitsandbytes:
  source .venv/bin/activate && \
  cd examples/qlora_single_gpu && \
  source ./bitsandbytes.sh

