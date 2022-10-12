from __future__ import annotations
from lun.data.metadata import load_images_using_metadata
from lun.data.images import load_image
from ldm.util import instantiate_from_config

import os
import torch

from random import shuffle

from typing import cast, Any, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms   # type: ignore

import glob
import json

import random

from omegaconf import OmegaConf

allowed_ext = ('png', 'jpg', 'jpeg', 'bmp')


class Local(Dataset):
    images: list[dict[str, Any]] | None
    crop: bool
    size: int
    flip: Any
    interpolation: Any
    limit_images: int | None

    def __init__(
        self,
        data_root: str = './danbooru-aesthetic',
        size: int = 512,
        interpolation: str = 'bicubic',
        flip_p: float | None = 0.5,
        crop: bool = True,
        metadata_params: dict[str, Any] | Any | None = None,
        limit_images: int | None = None,
    ):
        super().__init__()

        self.crop = crop

        self.data_root = data_root
        self.metadata_params = metadata_params
        self.size = size
        self.interpolation = {
            'linear': Image.LINEAR,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
        }[interpolation]
        self.images = None
        self.flip = None
        self.limit_images = limit_images
        if flip_p and flip_p > 0:
            self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def on_setup(self):
        print(
            f'{__package__}.{__name__} on_setup() for data_root={self.data_root} size={self.size}'
        )

        if self.metadata_params:
            metadata_params_converted = cast(
                dict,
                OmegaConf.to_container(self.metadata_params, resolve=True),
            )
            assert isinstance(metadata_params_converted, dict)
            for key in ('caption_sorter', 'custom_filter'):
                if key in metadata_params_converted:
                    metadata_params_converted[
                        key
                    ] = instantiate_from_config(
                        metadata_params_converted[key]
                    )
            self.images = load_images_using_metadata(
                self.data_root, **metadata_params_converted
            )
        else:
            self.images = self.load_images(self.data_root)

        print(f'Loaded {len(self.images)} captioned images')
        if self.limit_images:
            import random
            random.Random(1).shuffle(self.images)
            print(f'Limiting to {self.limit_images}')
            self.images = self.images[:self.limit_images]

    def load_images(
        self, data_root: str, consider_every_nth: int | None = None
    ) -> List[dict[str, Any]]:
        image_files: List[str] = []
        for e in allowed_ext:
            image_files.extend(glob.glob(f'{data_root}/img/' + '*.' + e))
        if consider_every_nth:
            image_files = image_files[0::consider_every_nth]

        print('Constructing image-caption map.')

        examples: List[dict] = []
        for i in image_files:
            hash = i[len(f'{data_root}/img/') :].split('.')[0]
            examples.append(
                {'image': i, 'text': f'{data_root}/txt/{hash}.txt'}
            )
        return examples

    def get_caption(self, i: int) -> str:
        example = self.images[i]
        if 'caption' in example:
            return example['caption']
        caption = open(example['text'], 'r').read()
        caption = caption.replace('  ', ' ').replace('\n', ' ').strip()
        return caption

    def __len__(self) -> int:
        if not self.images:
            self.on_setup()
        return len(self.images)

    def __getitem__(self, i: int) -> dict[str, Any]:
        if not self.images:
            self.on_setup()
        try:
            caption = self.get_caption(i)
        except Exception as e:
            print(
                f'Error with caption of {i} = {repr(self.images[i])} -- skipping',
                type(e),
                e,
            )
            raise

        image_file = self.images[i]['image']
        image = load_image(
            image_file, self.crop, self.size, self.interpolation, self.flip
        )
        return {
            'image': image,
            'caption': caption,
        }
