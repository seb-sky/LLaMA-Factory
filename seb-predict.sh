run_name=seb-"$(date -u +'%FT%H%MZ')"
out_dir=tmp/predict/"$run_name"
mkdir -p "$out_dir"

cp "$BASH_SOURCE" "$out_dir"/script.sh

python src/train_bash.py \
  --adapter_name_or_path /home/vauclasn/LLaMA-Factory/tmp/out/seb-2024-03-13T0857Z/output \
  --bf16 \
  --dataset skyguidetest \
  --dataset_dir /home/vauclasn/parse_skyguidetest/output/dataset \
  --ddp_find_unused_parameters False \
  --do_predict \
  --flash_attn \
  --lora_target q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj,o_proj \
  --max_samples 100 \
  --model_name_or_path google/gemma-7b-it \
  --output_dir "$out_dir"/output \
  --predict_with_generate \
  --quantization_bit 4 \
  --template gemma \
  --upcast_layernorm \
  --use_unsloth \
  |& tee "$out_dir"/log.txt
