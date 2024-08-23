DATA_PATH=./data_matching
CKPT_PATH=./ckpts
python -m torch.distributed.launch --nproc_per_node=1 \
main_task_rematching.py --do_rematching --num_thread_reader=4 \
--real_data_path ${DATA_PATH}/real_data \
--real_features_path ${DATA_PATH}/real_video \
--synthetic_data_path ${DATA_PATH}/synthetic_data \
--synthetic_features_path ${DATA_PATH}/synthetic_video \
--output_dir ${CKPT_PATH} \
--max_words 64 --max_frames 12 --batch_size_val 64 \
--datatype disaster \
--feature_framerate 1 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
