nproc_per_node=8

MAX_PIXELS=802816 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /path/to/Qwen2.5-VL-3b/ \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset data/imagenet_cold_sft.json \
    --torch_dtype bfloat16 \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs 1 \
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
    --output_dir output/cold-img-3b \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 0 \
    --attn_impl flash_attn \
    --deepspeed zero2
    