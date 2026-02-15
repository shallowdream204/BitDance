import contextlib
import io
import math
import os
import pickle
import tarfile
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return

    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64

    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def build_flat_index(outer_path: str, idx_path: str):
    if os.path.exists(idx_path):
        print(f"Index file {idx_path} already exists. Skipping index building.")
        return pickle.load(open(idx_path, "rb"))
    entries = []  # (offset, size, label)
    cats = set()
    idx = 0
    with tarfile.open(outer_path, "r:") as outer:
        for sub in outer.getmembers():
            if not sub.isfile() or not sub.name.endswith(".tar"):
                continue
            outer_off = sub.offset_data
            sub_fobj = outer.extractfile(sub)
            with tarfile.open(fileobj=sub_fobj, mode="r:") as inner:
                for m in inner.getmembers():
                    if not m.isfile():
                        continue
                    cat = m.name.split("_", 1)[0]
                    cats.add(cat)
                    abs_off = outer_off + m.offset_data
                    entries.append((abs_off, m.size, cat))
                    if idx % 1000 == 1:
                        print(idx, m.name, abs_off, m.size, cat)
                    idx += 1
    sorted_cats = sorted(cats)
    cat2idx = {c: i for i, c in enumerate(sorted_cats)}

    flat = [(off, size, cat2idx[c]) for off, size, c in entries]

    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
    with open(idx_path, "wb") as f:
        pickle.dump(
            flat,
            f,
        )
    print(f"Built flat index with {len(flat)} images.")
    return flat


class ImageNetTarDataset(Dataset):
    """
    ImageNet dataset stored in a tar file, avoid to decompress the whole dataset.
    You can direct use the original downloaded tar file (ILSVRC2012_img_train.tar) from official ImageNet website.
    The best practice is to copy the tar file to node's local disk or ramdisk (like /dev/shm/) first, to avoid remote I/O bottleneck.
    """

    def __init__(
        self,
        tar_file,
    ):
        self.tar_file = tar_file
        self.tar_handle = None
        self.files = build_flat_index(tar_file, tar_file + ".index")
        self.num_examples = len(self.files)

    def __len__(self):
        return self.num_examples

    def get_raw_image(self, index):
        if self.tar_handle is None:
            self.tar_handle = open(self.tar_file, "rb")

        offset, size, label = self.files[index]
        self.tar_handle.seek(offset)
        data = self.tar_handle.read(size)
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return image, label

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        return self.get_raw_image(idx)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def numpy_randrange(start, end):
    return int(np.random.randint(start, end))


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = numpy_randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = numpy_randrange(0, arr.shape[0] - image_size + 1)
    crop_x = numpy_randrange(0, arr.shape[1] - image_size + 1)
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def crop(pil_image, left, top, right, bottom):
    """
    Crop the image to the specified box.
    """
    return pil_image.crop((left, top, right, bottom))


class ImageCropDataset(Dataset):

    def __init__(
        self,
        raw_dataset,
        resolution,
        patch_size,
        seed=42,
    ):
        self.raw_dataset = raw_dataset
        self.resolution = resolution
        self.patch_size = patch_size
        self.aug_ratio = 1.0
        self.seed = seed
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_aug_ratio(self, aug_ratio):
        self.aug_ratio = aug_ratio

    def __len__(self):
        return len(self.raw_dataset)

    def crop_and_flip(self, image):
        is_aug = np.random.rand() < self.aug_ratio
        if not is_aug:
            image = center_crop_arr(image, self.resolution)
        else:
            image = random_crop_arr(image, self.resolution)

        arr = np.asarray(image)

        is_flip = int(np.random.randint(0, 2))
        if is_flip == 1:
            # horizontal flip
            arr = arr[:, ::-1, :]

        return arr.transpose(2, 0, 1)  # HWC to CHW

    def __getitem__(self, idx):
        with numpy_seed(self.seed, self.epoch, idx):
            image, label = self.raw_dataset[idx]
            samples = self.crop_and_flip(image)
            # to [-1, 1]
            samples = (samples.astype(np.float32) / 255.0 - 0.5) * 2.0
            samples = torch.from_numpy(samples).float()
            return (
                samples,
                torch.tensor(label).long(),
            )


def build_dataset(args):
    # use tarred imagenet dataset if data_path ends with .tar
    raw_dataset = (
        ImageNetTarDataset(args.data_path)
        if args.data_path.endswith(".tar")
        else ImageFolder(args.data_path)
    )
    return ImageCropDataset(
        raw_dataset,
        args.image_size,
        args.patch_size,
        seed=args.global_seed if hasattr(args, "global_seed") else 42,
    )