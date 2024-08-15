import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_disaster_retrieval import Disaster_DataLoader

def dataloader_disaster_train(args, tokenizer):
    disaster_dataset = Disaster_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(disaster_dataset)
    dataloader = DataLoader(
        disaster_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(disaster_dataset), train_sampler

def dataloader_disaster_test(args, tokenizer, subset="test"):
    disaster_testset = Disaster_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
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
DATALOADER_DICT["disaster"] = {"train":dataloader_disaster_train, "val":dataloader_disaster_test, "test":dataloader_disaster_test}
