accelerate launch \
  --config_file fsdp_config.yaml \
  train.py \
  --local_dir local/qwen3-4b \
  --input_model_filename Qwen/Qwen3-4B \
  --output_model_filename Qwen/Qwen3-4B-4bit \
  --train_data_local_path local/qwen3_4b_dataset \
  --qat True \
  --nbits 4 \
  --model_max_length 16384 \
  \
  --do_train False \
  --do_eval False \
  --fp16 False \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing False \
  \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 100 \
  --logging_strategy steps \
  --logging_steps 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --lr_scheduler_type cosine \
  --num_train_epochs 0 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --report_to tensorboard \
  --logging_dir local/output/runs/current \
  --disable_tqdm False
