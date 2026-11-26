import glob
import os

from datasets import load_dataset, load_from_disk, Features, Array3D
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image
import torch

HF_REPO = "huggan/flowers-102-categories"
HF_REVISION = "e8e6f53dc08d354a409c003af6e6df5fba2b293c"
SPLIT_SEED = 0
# Lab: L in [0, 100] -> [-0.5, 0.5] (same range as the HSV V input); a, b scaled to about [-1, 1].
AB_SCALE = 110.0


def to_hsv(image):
    hsv = np.array(image.convert("HSV"))
    hs = hsv[:,:,0:-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    v =  hsv[:,:,-1][None,].astype(np.float32) / 255.0 - 0.5
    return hs, v


def to_lab(image):
    from skimage.color import rgb2lab
    lab = rgb2lab(np.asarray(image.convert("RGB"))).astype(np.float32)
    ab = lab[:, :, 1:].transpose(2, 0, 1) / AB_SCALE
    l = lab[:, :, 0][None,] / 100.0 - 0.5
    return ab, l


def convert(image, color_space):
    """PIL RGB image -> (target [2,H,W], input [1,H,W]) float32 arrays."""
    return to_lab(image) if color_space == "lab" else to_hsv(image)


def to_rgb(inp, target, color_space):
    """Inverse of convert: tensors/arrays [1,H,W] input and [2,H,W] target -> uint8 RGB [H,W,3]."""
    inp = np.asarray(inp, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if color_space == "lab":
        from skimage.color import lab2rgb
        lab = np.concatenate([(inp + 0.5) * 100.0, target * AB_SCALE], axis=0).transpose(1, 2, 0)
        return (np.clip(lab2rgb(lab), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    hsv = np.concatenate([target, inp + 0.5], axis=0)
    hsv = (np.clip(hsv, 0, 1) * 255.0).astype(np.uint8).transpose(1, 2, 0)
    return np.array(Image.fromarray(hsv, mode="HSV").convert("RGB"))


class TrainDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, color_space="hsv"):
        self.color_space = color_space
        def map_func(examples):
            examples["image"] = np.array(examples["image"].convert("RGB").resize((96,96)))
            return examples
        self.hf_dataset = hf_dataset.map(map_func, batched=False, features=Features({
            "image": Array3D(dtype="float32", shape=(96, 96, 3)),
        })).with_format("numpy")
        self.transforms = A.Compose([
            A.RandomScale(),
            A.Rotate(20),
            A.RandomCrop(64, 64),
            A.HorizontalFlip(p=0.5),
            A.Blur(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]["image"]
        image /= 255.0
        image = self.transforms(image=image)["image"]
        image *= 255.0
        target, inp = convert(Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)), self.color_space)
        return torch.from_numpy(inp), torch.from_numpy(target)

class ValDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, color_space="hsv"):
        def map_func(examples):
            image = examples["image"].convert("RGB")
            target, inp = convert(image.resize((64,64)), color_space)
            examples["hs"] = target
            examples["v"] = inp
            return examples
        self.hf_dataset = hf_dataset.map(map_func, remove_columns=["image"], batched=False, features=Features({
            "hs": Array3D(dtype="float32", shape=(2, 64, 64)),
            "v": Array3D(dtype="float32", shape=(1, 64, 64)),
        })).with_format("torch")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return item["v"], item["hs"]


def load_split(data_dir=None):
    """Train/test split with a fixed seed, so every experiment sees the same validation images.

    data_dir: directory holding the HF dataset snapshot (what `r2.py pull-dataset hf:...` writes).
    Without it: the old ./flowers-102-split from generate_split.py if present, else the pinned
    revision is downloaded to data/flowers-102-categories.
    """
    if not data_dir and os.path.isdir("./flowers-102-split"):
        return load_from_disk("./flowers-102-split")
    if not data_dir:
        # Same files and order as `r2.py pull-dataset hf:<HF_REPO>@<HF_REVISION> data` on the Pod.
        from huggingface_hub import snapshot_download
        data_dir = snapshot_download(repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION,
                                     local_dir="data/flowers-102-categories")
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {data_dir}")
    ds = load_dataset("parquet", data_files=files, split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c != "image"])
    return ds.train_test_split(test_size=0.1, seed=SPLIT_SEED)


def get_dataset(color_space="hsv", data_dir=None, limit=None):
    dataset = load_split(data_dir)
    train, test = dataset["train"], dataset["test"]
    if limit:
        train, test = train.select(range(min(limit, len(train)))), test.select(range(min(limit, len(test))))
    train_dataset = TrainDatasetWrapper(train, color_space)
    val_dataset = ValDatasetWrapper(test, color_space)
    return train_dataset, val_dataset
