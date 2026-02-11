
MAX_PIXELS=802816 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /path/to/cold-img-3b/checkpoint-25/ \
    --model_type qwen2_5_vl \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --dataset data/imagenet_left_grpo.json \
    --external_plugins examples/train/my/plugin/plugin.py \
    --reward_funcs external_exact_acc \
    --use_vllm false \
    --torch_dtype bfloat16 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --split_dataset_ratio 0 \
    --learning_rate 1e-5 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/qwen2_5_vl-img-grpo-3b \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 0 \
    --max_completion_length 128 \
    --num_generations 8 \
    --temperature 0.9 \
    --top_p 1.0 \
    --beta 0.04 \
    --log_completions true \
    --attn_impl flash_attn \
    --deepspeed zero2
