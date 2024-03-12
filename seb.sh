run_name=seb-"$(date -u +'%FT%H%MZ')"

python src/train_bash.py \
  --model_name_or_path google/gemma-7b-it \
  --quantization_bit 4 \
  --flash_attn \
  --use_unsloth \
  --upcast_layernorm
  --export_dir tmp/out/export \
  --template gemma \
  --dataset ultra_chat \
  --dataset_dir data \
  --overwrite_cache \
  --max_samples 1000 \
  --val_size 0.1 \
  --output_dir tmp/out/output/"$run_name" \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --logging_dir tmp/out/tensorboard/"$run_name" \
  --logging_steps 0.01 \
  --save_strategy steps \
  --save_steps 0.1 \
  --bf16 \
  --eval_steps 0.1 \
  --load_best_model_at_end \
  --report_to tensorboard \
  --ddp_find_unused_parameters False \
  --lora_target q_proj,v_proj \


