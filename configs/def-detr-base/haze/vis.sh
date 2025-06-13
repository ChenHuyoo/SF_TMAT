BATCH_SIZE=2
DATA_ROOT=/run/media/test/desk2//MRT-release-main/data/
OUTPUT_DIR=./outputs/def-detr-base/city2foggy/vis

CUDA_VISIBLE_DEVICES=1 python -u main.py \
--backbone resnet50 \
--num_encoder_layers 6 \
--num_decoder_layers 6 \
--num_classes 4 \
--data_root ${DATA_ROOT} \
--source_dataset cityscapes \
--target_dataset foggy_cityscapes \
--eval_batch_size ${BATCH_SIZE} \
--mode eval \
--output_dir ${OUTPUT_DIR} \
--resume ${OUTPUT_DIR}/../source_only/model_best.pth \
