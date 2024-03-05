from datasets import load_from_disk, Features, Array3D
import numpy as np
from torch.utils.data import Dataset

def transforms(examples):
    image = examples["image"]
    hsv = np.array(image.resize((64,64)).convert("HSV"))
    examples["hs"] = hsv[:,:,0:-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    examples["v"] =  hsv[:,:,-1][None,].astype(np.float32) / 255.0 - 0.5
    return examples

class HfDatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return (item['v'], item['hs'])

def get_dataset():
    features = Features({
        "hs": Array3D(dtype="float32", shape=(2, 64, 64)),
        "v": Array3D(dtype="float32", shape=(1, 64, 64)),
    })

    dataset = load_from_disk("./flowers-102-split")
    dataset = dataset.map(transforms, remove_columns=["image"], batched=False, features=features).with_format("torch")
    return HfDatasetWrapper(dataset["train"]), HfDatasetWrapper(dataset["test"])
