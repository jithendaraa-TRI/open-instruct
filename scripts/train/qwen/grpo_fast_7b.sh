export UV_PROJECT_ENVIRONMENT=/home/ubuntu/.venv
export UV_CACHE_DIR=/mnt/efs/fs1/jith/open-instruct/.uv_cache
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HF_HOME=/mnt/efs/fs1/huggingface-cache/
export RAY_ADDRESS="10.163.137.252:8888"
# Note: Don't set RAY_TEMP_DIR to EFS - sockets don't work on network filesystems.
# Start Ray with object spilling to EFS instead (see bottom of file).

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
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,gsm8k::hamish_zs_reasoning,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning \
    --oe_eval_max_length 8192 \
    --temperature 1.0 \
    --total_episodes 5000000 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 1 \
    --num_learners_per_node 4 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 12 \
    --lr_scheduler_type linear \
    --seed 1 \
    --local_eval_every 30 \
    --save_freq 40 \
    --vllm_sync_backend nccl \
    --vllm_enforce_eager \
    --gradient_checkpointing \
    --with_tracking \
    --wandb_project_name open-instruct-grpo-fast-qwen-7b

# ===== RAY CLUSTER SETUP (run these BEFORE the training script) =====
# Head (10.163.137.252):
#   uv run ray stop --force && uv run ray start --head --port=8888 --dashboard-host=0.0.0.0 --object-spilling-directory=/mnt/efs/fs1/jith/ray_spill
# Worker (other nodes):
#   uv run ray stop --force && uv run ray start --address="10.163.137.252:8888" --dashboard-host=0.0.0.0 --object-spilling-directory=/mnt/efs/fs1/jith/ray_spill
# Verify: uv run ray status