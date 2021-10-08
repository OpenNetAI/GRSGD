import torch
import torch.distributed as dist

from datetime import timedelta


def ring_reduce(model, rank, args):
    for p in model.parameters():
        if p.requires_grad:
            dist.all_reduce(p.data)
            p.data /= args.world_size

__all__ = ['ring_reduce']
