export UV_PROJECT_ENVIRONMENT=/mnt/efs/fs1/jith/open-instruct/.venv
export UV_CACHE_DIR=/mnt/efs/fs1/jith/open-instruct/.uv_cache
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HF_HOME=/mnt/efs/fs1/huggingface-cache/

uv run python open_instruct/grpo_fast.py \
    --exp_name qwen2.5_7b_grpo_fast_zero \
    --beta 0.0 \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --kl_estimator 2 \
    --learning_rate 5e-7 \
    --dataset_mixer_list ai2-adapt-dev/math_ground_truth_zs 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list ai2-adapt-dev/math_ground_truth_zs 16 \
    --dataset_mixer_eval_list_splits train \
    --max_prompt_token_length 2048 \
    --response_length 4096 \
    --pack_length 6144 \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --stop_strings "</answer>" \
    --apply_r1_style_format_reward \
    --apply_verifiable_reward true \
    --non_stop_penalty \
    --non_stop_penalty_value 0.0 \
    --chat_template_name r1_simple_chat_postpend_think \
    --temperature 1.0 \
    --total_episodes 5000000 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 2 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 2 \
    --lr_scheduler_type linear \
    --seed 1 \
    --local_eval_every 30 \
    --save_freq 40 \
    --vllm_sync_backend nccl \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    # --with_tracking