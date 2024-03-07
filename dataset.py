from datasets import load_from_disk, Features, Array3D
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from PIL import Image
import torch


def to_hsv(image):
    hsv = np.array(image.convert("HSV"))
    hs = hsv[:,:,0:-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    v =  hsv[:,:,-1][None,].astype(np.float32) / 255.0 - 0.5
    return hs, v
    

class TrainDatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        def map_func(examples):
            examples["image"] = np.array(examples["image"].resize((96,96)))
            return examples
        self.hf_dataset = hf_dataset.map(map_func, batched=False, features=Features({
            "image": Array3D(dtype="float32", shape=(96, 96, 3)),
        })).with_format("numpy")
        self.transforms = A.Compose([
            A.RandomCrop(64, 64),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        image = self.hf_dataset[idx]["image"]
        image = self.transforms(image=image)["image"]
        hs, v = to_hsv(Image.fromarray(image.astype(np.uint8)))
        return torch.from_numpy(v), torch.from_numpy(hs)

class ValDatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        def map_func(examples):
            image = examples["image"]
            hs, v = to_hsv(image.resize((64,64)))
            examples["hs"] = hs
            examples["v"] =  v
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


def get_dataset():
    dataset = load_from_disk("./flowers-102-split")
    train_dataset = TrainDatasetWrapper(dataset["train"])
    val_dataset = ValDatasetWrapper(dataset["test"])
    return train_dataset, val_dataset
