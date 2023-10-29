import copy
import bisect
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from lib.dataset.svol_dataset import build_dataset, collate_fn
from lib.dataset.sampler import GroupedBatchSampler, IterationBasedBatchSampler
from lib.utils.comm import get_world_size, get_rank


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        vid_info = dataset.get_vid_info(i)
        aspect_ratio = float(vid_info["height"]) / float(vid_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(dataset, sampler, aspect_grouping, videos_per_batch, num_iters=None, start_iter=0):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = GroupedBatchSampler(
            sampler, group_ids, videos_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = BatchSampler(
            sampler, videos_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def build_dataloader(args, phase="train", distributed=False, start_iter=0):
    num_gpus = get_world_size()
    dataset = build_dataset(args, phase)
    sampler = make_data_sampler(
        dataset,
        shuffle=True if phase == "train" else False,
        distributed=distributed
    )
    videos_per_batch = args.bs if phase == "train" else args.eval_bs
    if get_rank() == 0:
        print(f"phase: {phase}, videos_per_batch: {videos_per_batch}, num_gpus: {num_gpus}")
    videos_per_gpu = videos_per_batch // num_gpus if phase == "train" else videos_per_batch
    start_iter = start_iter if phase == "train" else 0
    num_iters = args.num_iters if phase == "train" else None
    aspect_grouping = [1] if args.aspect_ratio_grouping else []
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, aspect_grouping, videos_per_gpu, num_iters, start_iter
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory
    )
    return dataloader
