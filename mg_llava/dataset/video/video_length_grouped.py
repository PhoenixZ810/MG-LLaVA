# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Optional, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Sampler


def get_length_grouped_indices(lengths, group_batch_size, generator=None):

    def process(lengths, group_batch_size, generator=None):
        indices = torch.randperm(len(lengths), generator=generator)
        megabatches = [
            indices[i : i + group_batch_size].tolist() for i in range(0, len(lengths), group_batch_size)
        ]
        megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
        return megabatches

    assert all(leng != 0 for leng in lengths), 'Should not have zero length.'
    if all(leng > 100000 for leng in lengths):
        # all videos
        lengths = [i-100000 for i in lengths]
        megabatches = process(lengths, group_batch_size, generator=generator)
    elif all(leng > 0 for leng in lengths) and not all(leng < 100000 for leng in lengths):
        # image + video, no text
        mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if (l > 0 and l < 100000)])
        video_indices, video_lengths = zip(*[(i, l-100000) for i, l in enumerate(lengths) if l > 100000])
        mm_megabatches = []
        for mm_megabatch in process(mm_lengths, group_batch_size, generator=generator):
            mm_megabatches.append([mm_indices[i] for i in mm_megabatch])
        video_megabatches = []
        for video_megabatch in process(video_lengths, group_batch_size, generator=generator):
            video_megabatches.append([video_indices[i] for i in video_megabatch])
        last_mm = mm_megabatches[-1]
        last_video = video_megabatches[-1]
        last_batch = last_mm + last_video
        megabatches = mm_megabatches[:-1] + video_megabatches[:-1]

        megabatch_indices = torch.randperm(len(megabatches), generator=generator)
        megabatches = [megabatches[i] for i in megabatch_indices]

        if len(last_batch) > 0:
            # megabatches.append(sorted(last_batch, key=lambda i: abs(lengths[i]), reverse=True))
            megabatches.append(last_batch)
    elif all(leng > 0 for leng in lengths) or all(leng < 0 for leng in lengths):
        # all images or all text
        megabatches = process(lengths, group_batch_size, generator=generator)
    else:
        # image, video, text
        mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if (l > 0 and l<100000)])
        video_indices, video_lengths = zip(*[(i, l-100000) for i, l in enumerate(lengths) if l>100000])
        lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])
        mm_megabatches = []
        for mm_megabatch in process(mm_lengths, group_batch_size, generator=generator):
            mm_megabatches.append([mm_indices[i] for i in mm_megabatch])
        video_megabatches = []
        for video_megabatch in process(video_lengths, group_batch_size, generator=generator):
            video_megabatches.append([video_indices[i] for i in video_megabatch])
        lang_megabatches = []
        for lang_megabatch in process(lang_lengths, group_batch_size, generator=generator):
            lang_megabatches.append([lang_indices[i] for i in lang_megabatch])

        last_mm = mm_megabatches[-1]
        last_video = video_megabatches[-1]
        last_lang = lang_megabatches[-1]
        last_batch = last_mm + last_video + last_lang
        megabatches = mm_megabatches[:-1] + video_megabatches[:-1] + lang_megabatches[:-1]

        megabatch_indices = torch.randperm(len(megabatches), generator=generator)
        megabatches = [megabatches[i] for i in megabatch_indices]

        if len(last_batch) > 0:
            # megabatches.append(
            #     sorted(
            #         last_batch,
            #         key=lambda i: (
            #             abs(lengths[i])
            #             if abs(lengths[i]) < 100000
            #             else abs(lengths[i]) - 100000
            #         ),
            #         reverse=True,
            #     )
            # )
            megabatches.append(last_batch)

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length,
    # the longest element is the first
    megabatch_maximums = [
        (
            abs(lengths[megabatch[0]])
            if abs(lengths[megabatch[0]]) < 100000
            else abs(lengths[megabatch[0]]) - 100000
        )
        for megabatch in megabatches
    ]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


class VideoLengthGroupedSampler(Sampler):

    def __init__(
        self,
        dataset: Sized,
        per_device_batch_size: int,
        length_property='length',
        mega_batch_mult: Optional[int] = None,
        seed: Optional[int] = None,
        round_up: bool = True,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            num_iters = math.ceil(len(self.dataset) / world_size / per_device_batch_size)
            self.num_samples = num_iters * per_device_batch_size
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

        total_batch_size = per_device_batch_size * self.world_size
        if mega_batch_mult is None:
            # Default for mega_batch_mult: 50 or the number to get 4
            # megabatches, whichever is smaller.
            mega_batch_mult = min(len(self.dataset) // (total_batch_size * 4), 50)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1
        self.group_batch_size = mega_batch_mult * total_batch_size

        if isinstance(self.dataset, TorchConcatDataset):
            length = []
            for sub_dataset in self.dataset.datasets:
                length.extend(getattr(sub_dataset, length_property))
            self.length = length
        else:
            self.length = getattr(self.dataset, length_property)
        assert isinstance(self.length, (list, tuple))

        self.total_batch_size = total_batch_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(
            lengths=self.length, group_batch_size=self.group_batch_size, generator=generator
        )
        assert len(set(indices)) == len(indices)
        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[: self.total_size]
        # subsample
        assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_samples
        return iter(indices)

    # def __len__(self) -> int:
    #     """The number of samples in this rank."""
    #     return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
