
MAX_PIXELS=802816 \
NPROC_PER_NODE=8 \
swift sft \
    --model /path/to/cold-img-3b/checkpoint-25/ \
    --model_type qwen2_5_vl \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset data/imagenet_left_sft.json \
    --torch_dtype bfloat16 \
    --freeze_vit false \
    --freeze_aligner false \
    --max_steps 600 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --split_dataset_ratio 0 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 128 \
    --output_dir output/qwen2_5_vl-img-sft-3b \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 0 \
    --attn_impl flash_attn \
    --deepspeed zero2
    