run_name=seb-"$(date -u +'%FT%H%MZ')"
out_dir=tmp/predict/"$run_name"
mkdir -p "$out_dir"

cp "$BASH_SOURCE" "$out_dir"/script.sh

python src/train_bash.py \
  --dataset skyguidetest \
  --dataset_dir /home/vauclasn/parse_skyguidetest/output/dataset \
  --do_predict \
  --model_name_or_path google/gemma-7b-it \
  --adapter_name_or_path /home/vauclasn/LLaMA-Factory/tmp/out/seb-2024-03-13T0857Z/output \
  --output_dir "$out_dir"/output \
  --template gemma \
  --predict_with_generate \
  |& tee "$out_dir"/log.txt
