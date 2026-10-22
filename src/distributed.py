import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def initialize(requested):
    device = torch.device(requested)
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if device.type == "cuda":
        device = torch.device("cuda", local if world > 1 else (device.index or 0))
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return device, rank, world


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap(model, device, world):
    if world == 1:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def broadcast(value, world):
    if world > 1:
        values = [value]
        dist.broadcast_object_list(values, src=0)
        value = values[0]
    return value


def gather(value, world):
    values = [value]
    if world > 1:
        values = [None] * world
        dist.all_gather_object(values, value)
    return values


def finish():
    if dist.is_initialized():
        dist.destroy_process_group()
