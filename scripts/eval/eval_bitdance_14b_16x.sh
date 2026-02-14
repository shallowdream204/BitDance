CKPT_PATH=models/BitDance-14B-16x

# DPG
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    eval/eval_dpg.py \
    --model_path ${CKPT_PATH} \
    --data_path eval/dpg_bench/prompts.json \
    --save_dir results/dpg \
    --image_size 1024 1024 \
    --guidance_scale 7.5 \
    --num_sampling_steps 50

# GenEval
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    eval/eval_geneval.py \
    --model_path ${CKPT_PATH} \
    --data_path eval/geneval/prompts/evaluation_metadata_long.jsonl \
    --save_dir results/geneval \
    --image_size 1024 1024 \
    --guidance_scale 13.0 \
    --num_sampling_steps 50