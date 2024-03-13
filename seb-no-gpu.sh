run_name=seb-"$(date -u +'%FT%H%MZ')"

export CUDA_VISIBLE_DEVICES=-1

python src/train_bash.py \
  --bf16 \
  --dataset skyguidetest \
  --dataset_dir /home/vauclasn/parse_skyguidetest/output/dataset \
  --ddp_find_unused_parameters False \
  --do_eval \
  --do_train \
  --eval_steps 0.1 \
  --evaluation_strategy steps \
  --export_dir tmp/out/export \
  --load_best_model_at_end \
  --logging_dir tmp/out/tensorboard/"$run_name" \
  --logging_steps 0.01 \
  --max_samples 10000 \
  --model_name_or_path google/gemma-7b-it \
  --output_dir tmp/out/output/"$run_name" \
  --overwrite_cache \
  --overwrite_output_dir \
  --report_to tensorboard \
  --save_steps 0.1 \
  --save_strategy steps \
  --template gemma \
  --finetuning_type full \
  --val_size 0.1
