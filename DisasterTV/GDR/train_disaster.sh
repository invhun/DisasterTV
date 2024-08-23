DATA_PATH=./data_GDR
CKPT_PATH=./ckpts
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_retrieval.py --do_train --num_thread_reader=4 \
--epochs=5 --batch_size=64 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/grid_image \
--output_dir ${CKPT_PATH} \
--lr 3.1e-4 --max_words 64 --batch_size_val 64 \
--datatype disaster --coef_lr 5.75e-4 \
--freeze_layer_num 0 \
--loose_type --linear_patch 2d \
--pretrained_clip_name ViT-B/32 \
