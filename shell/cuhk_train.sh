export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=1

OUTPUT_DIR="output/train/XXX/XXX"

torchrun --rdzv_endpoint=127.0.0.1:29501 --nproc_per_node=1 \
train_ps.py \
--config ./configs/blip_gmm.yaml \
--output_dir $OUTPUT_DIR \