run_name=seb-"$(date -u +'%FT%H%MZ')"
out_dir=tmp/out/"$run_name"
mkdir -p "$out_dir"

cp "$BASH_SOURCE" "$out_dir"/script.sh

python src/train_bash.py \
  --bf16 \
  --dataset skyguidetest \
  --dataset_dir /home/vauclasn/parse_skyguidetest/output/dataset \
  --ddp_find_unused_parameters False \
  --do_eval \
  --do_train \
  --eval_steps 0.1 \
  --evaluation_strategy steps \
  --export_dir "$out_dir"/export \
  --flash_attn \
  --load_best_model_at_end \
  --logging_dir "$out_dir" \
  --logging_steps 0.01 \
  --lora_target q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj,o_proj \
  --model_name_or_path google/gemma-7b-it \
  --num_train_epochs 10 \
  --output_dir "$out_dir"/output \
  --overwrite_cache \
  --overwrite_output_dir \
  --quantization_bit 4 \
  --report_to tensorboard \
  --save_steps 0.1 \
  --save_strategy steps \
  --template gemma \
  --upcast_layernorm \
  --use_unsloth \
  --val_size 0.1 \
  |& tee "$out_dir"/log.txt
