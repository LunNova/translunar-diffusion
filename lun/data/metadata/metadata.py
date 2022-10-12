from __future__ import annotations
from typing import Any, List, Set, Callable
import json
import glob
import random
import os

from ldm.util import instantiate_from_config

# This is in its own module so it can be compiled with mypyc

allowed_ext = ('png', 'jpg', 'jpeg', 'bmp')

# Yes I am loading every single json file in separately
# It's fast enough to not be worth optimizing :L
def load_images_using_metadata(
    data_root: str,
    consider_every_nth: int | None = None,
    score_tags: List[int] = [],
    shuffle_tag_p: float | None = None,
    abs_min_score: int = 25,
    min_score: int = 50,
    blacklist_tags: List[str] | None = None,
    non_caption_tags: List[str] | Set[str] | None = None,
    non_caption_tag_prefixes: List[str] | None = None,
    replacements: dict[str, str] | None = None,
    tag_bonus_scores: dict[str, int] | None = None,
    skip_caption: bool = False,
    caption_sorter: Callable[[str], list] | None = None,
    custom_filter: Callable[[str], bool] | None = None,
    tag_separator: str = ', ',
    final_tag_separator: bool = False,
    aspect_range: List[int] = [0.71, 1.42],
) -> List[dict[str, Any]]:
    examples: List[dict[str, Any]] = []

    meta_glob = f'{data_root}/metadata/*/*.json'
    print(f'Loading data from {meta_glob}')
    meta_files = glob.glob(meta_glob)
    non_caption_tags_set = set()
    if non_caption_tags:
        non_caption_tags_set = set(non_caption_tags)
        print("Filtering out non_caption_tags:", non_caption_tags, "\nprefixes:", non_caption_tag_prefixes)

    aspect_min, aspect_max = aspect_range

    def filter_non_caption_tag(tag: str) -> bool:
        if tag in non_caption_tags_set:
            return False
        if non_caption_tag_prefixes and any(tag.startswith(prefix) for prefix in non_caption_tag_prefixes):
            return False
        return True

    if consider_every_nth:
        meta_files = meta_files[0::consider_every_nth]
    print(f'Considering {len(meta_files)} metadata files to find images')
    for meta_file in meta_files:
        meta = None
        with open(meta_file, 'r') as f:
            try:
                meta = json.load(f)
            except Exception as e:
                # print(f"File {f} failed due to {e}, skipping")
                pass
        if not meta:
            continue
        aspect_ratio = float(meta['aspect_ratio'])
        if not (0.71 <= aspect_ratio <= 1.42):
            continue
        score = int(meta['score'])
        if score < abs_min_score:
            continue
        path = f'{data_root}/{str(meta["path"])}'
        if not os.path.exists(path):
            print(f'Skipping missing image {path}')
            continue
        ext = os.path.splitext(path)[1][1:].lower()
        if ext not in allowed_ext:
            continue
        tags: List[str] = meta['tags']
        if custom_filter and not custom_filter(tags):
            continue
        for i in score_tags:
            if score >= i:
                tags.append(f'scr{i}')
                break
        if tag_bonus_scores:
            for k, v in tag_bonus_scores.items():
                if k in tags:
                    score += int(v)
        if score < min_score:
            continue
        if blacklist_tags and any(x for x in blacklist_tags if x in tags):
            continue
        caption = ''
        if not skip_caption:
            if non_caption_tags_set:
                tags = list(filter(filter_non_caption_tag, tags))

            if shuffle_tag_p and (random.random() < shuffle_tag_p):
                random.shuffle(tags)
            else:
                tags.sort(key=caption_sorter)

            if replacements:
                caption = '||'.join(tags) + (
                    '||' if final_tag_separator else ''
                )
                for rk, rv in replacements.items():
                    caption = caption.replace(rk, rv)
                caption = caption.replace('||', tag_separator)
            else:
                caption = tag_separator.join(tags) + (
                    tag_separator if final_tag_separator else ''
                )
        examples.append({'image': path, 'caption': caption})
    return examples
