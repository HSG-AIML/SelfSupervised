import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import random
import json
from selfsup.methods.simclr.datasets.transforms_image import GaussianBlur, GaussianBlur2dKornia
np.random.seed(0)


class COCO2014(Dataset):
    r"""Dataset class for MS COCO 2014."""
    def __init__(self, ds_path, transform, mode="train"):
        super().__init__()
        if mode == "train":
            self.image_folder = "train2014"
        elif mode == "test":
            self.image_folder = "test2014"
        elif mode == "val":
            self.image_folder = "val2014"
        self.ds_path = ds_path
        self.transform = transform

        with open(os.path.join(ds_path, "annotations/instances_val2014.json"), "r") as json_file:
            annotations = json.load(json_file) 
        self.filename_to_imgid = {a["file_name"]:int(a["id"]) for a in annotations["images"]} 
        self.imgid_to_catid = {int(a["image_id"]):a["category_id"] for a in annotations["annotations"]} 
        self.filenames = [f for f,id in self.filename_to_imgid.items() if id in self.imgid_to_catid.keys()]
        print(f"Number of files: {len(self.filenames)}")

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.ds_path, self.image_folder, filename)
        image = Image.open(image_path).convert('RGB')
        out = self.transform(image)

        lbl = self.imgid_to_catid[self.filename_to_imgid[filename]]

        return out, lbl
        
    def __len__(self):
        return len(self.filenames)


def _get_transforms(input_shape):
    r"""Transform for Inference"""
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape),
                                          transforms.ToTensor()])
    return data_transforms


def get_coco_dataloader(args):
    r"""Returns train and validation dataloaders."""
    data_transforms = _get_transforms(args["input_shape"])
    dataset = COCO2014(args["dataset_path"], 
                       transform=data_transforms,
                       mode="val")                           

    loader = DataLoader(dataset, 
                        batch_size=256, 
                        num_workers=5, 
                        drop_last=True)

    return loader
