nbits=4
size=4

accelerate launch \
  --config_file fsdp_config.yaml \
  train.py \
  --local_dir local/qwen3-${size}b \
  --input_model_filename Qwen/Qwen3-${size}B \
  --output_model_filename Qwen/Qwen3-${size}B-${nbits}bit \
  --train_data_local_path local/qwen3-${size}b-dataset \
  --qat True \
  --nbits $nbits \
  --group_size 128 \
  --model_max_length 16384 \
  \
  --do_train True \
  --do_eval False \
  --fp16 False \
  --bf16 True \
  --tf32 False \
  --gradient_checkpointing False \
  \
  --num_train_epochs 2 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_strategy epoch \
  --logging_strategy steps \
  --logging_steps 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --lr_scheduler_type cosine \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --report_to tensorboard \
  --logging_dir local/output/runs/current \
  --disable_tqdm False
