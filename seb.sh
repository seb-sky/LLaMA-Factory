run_name=seb-"$(date -u +'%FT%H%MZ')"

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
  --max_samples 1000 \
  --val_size 0.1 \
  --output_dir tmp/out/output/"$run_name" \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --logging_dir tmp/out/tensorboard/"$run_name" \
  --save_strategy steps \
  --bf16 \
  --eval_steps 0.1 \
  --save_steps 0.1 \
  --load_best_model_at_end \
  --report_to tensorboard \
  --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj


