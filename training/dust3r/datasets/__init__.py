"""Minimal dust3r.datasets package — only the 3 datasets needed for the
small-scale lingbot-map training test (VKitti2, WildRGBD, BlendedMVS).

The full LoG-VGGT version imports ~30 dataset classes; we keep the import
surface small so missing dependencies (h5py for blendedmvs is the only
non-stdlib one) don't block iteration.
"""

from .utils.transforms import *  # noqa: F401,F403
from .base.batched_sampler import BatchedRandomSampler  # noqa: F401

from .co3d import Co3d_Multi  # noqa: F401  (parent of WildRGBD_Multi)
from .blendedmvs import BlendedMVS_Multi  # noqa: F401
from .vkitti2 import VirtualKITTI2_Multi  # noqa: F401
from .wildrgbd import WildRGBD_Multi  # noqa: F401


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    fixed_length=False,
    world_size=1,
):
    """Build a DataLoader with the dataset's custom batched sampler if it
    exposes one (this is how dust3r-style datasets produce
    (seq_idx, num_views, aspect_ratio) tuples).

    `dataset` may be a Python expression string like
    ``"VirtualKITTI2_Multi(split='train', ROOT='/...', resolution=(518,392), num_views=4)"``
    — in that case it is ``eval``-ed against this module's namespace.
    """
    import torch

    if isinstance(dataset, str):
        dataset = eval(dataset)  # noqa: S307 — dust3r upstream uses the same pattern

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=world_size,
            fixed_length=fixed_length,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )
    except (AttributeError, NotImplementedError):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )
