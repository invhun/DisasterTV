import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_disaster_rematching import Disaster_Rematching_DataLoader



def dataloader_disaster_rematching(args, tokenizer, real_subset="real_test", synthetic_subset="synthetic_1261"):
    disaster_testset = Disaster_Rematching_DataLoader(
        real_subset=real_subset,
        synthetic_subset=synthetic_subset,
        real_data_path=args.real_data_path,
        real_features_path=args.real_features_path,
        synthetic_data_path=args.synthetic_data_path,
        synthetic_features_path=args.synthetic_features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_disaster = DataLoader(
        disaster_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_disaster, len(disaster_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["disaster"] = {"rematching":dataloader_disaster_rematching}