export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3

OUTPUT_DIR="output/eval/XXX/XXX"

torchrun --rdzv_endpoint=127.0.0.1:29506 --nproc_per_node=4 \
train_ps.py \
--config ./configs/blip_gmm.yaml \
--output_dir $OUTPUT_DIR \
--eval_mAP \
--evaluate
