import torch
from torch.utils import data

import torchtext

from torch._six import int_classes, string_classes
import collections
import re


def torchtext_collate(batch):
    r"""Slightly different from default_collate: add torchtext.data.Batch to it.
        Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: torchtext_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], torchtext.data.Batch):  # difference here
        return {key: torchtext_collate([getattr(d, key) for d in batch]) for key in batch[0].dataset.fields.keys()}
    elif isinstance(batch[0], torchtext.data.Example):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [torchtext_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))
