run_name=seb-"$(date -u +'%FT%H%MZ')"
out_dir=tmp/predict/"$run_name"
mkdir -p "$out_dir"

cp "$BASH_SOURCE" "$out_dir"/script.sh

python src/train_bash.py \
  --adapter_name_or_path /home/vauclasn/LLaMA-Factory/tmp/out/seb-2024-03-20T1217Z/output \
  --bf16 \
  --dataset skyguidetest_test \
  --dataset_dir /home/vauclasn/parse_skyguidetest/output/dataset/v1 \
  --ddp_find_unused_parameters False \
  --do_predict \
  --flash_attn \
  --lora_target q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj,o_proj \
  --model_name_or_path beowolx/CodeNinja-1.0-OpenChat-7B \
  --output_dir "$out_dir"/output \
  --predict_with_generate \
  --quantization_bit 4 \
  --template openchat \
  --upcast_layernorm \
  |& tee "$out_dir"/log.txt
