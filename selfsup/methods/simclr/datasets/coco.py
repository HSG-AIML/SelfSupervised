import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import datasets
import random
from selfsup.methods.simclr.datasets.transforms_image import GaussianBlur, GaussianBlur2dKornia
np.random.seed(0)



class COCO2014(Dataset):
    r"""Dataset class for MS COCO 2014."""
    def __init__(self, ds_path, transform, mode="train"):
        super().__init__()
        if mode == "train":
            folder_name = "train2014"
        elif mode == "test":
            folder_name = "test2014"
        elif mode == "val":
            folder_name = "val2014"
        self.transform = transform
        self.all_image_paths = glob.glob(os.path.join(ds_path, folder_name, "*.jpg"))
        if len(self.all_image_paths) == 0:
            raise RuntimeError(f"Referring to empty folder.")

    def __getitem__(self, idx):
        image = Image.open(self.all_image_paths[idx]).convert('RGB')
        out = self.transform(image)
        return out
        
    def __len__(self):
        return len(self.all_image_paths)


class SimCLRDualTransform(object):
    r"""Transform class for SimCLR.
    Returns two version of the transform on the same input.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        # Apply first transformation
        xi = self.transform(sample)
        # Gaussian blur
        if random.random() > 0.5:
            sigma = 0.01 + (random.random() * 2)
            kornia_gauss = GaussianBlur2dKornia(kernel_size=(5, 5), sigma=(sigma, sigma))
            xi = kornia_gauss(xi.unsqueeze(0)).squeeze(0)
        
        # Apply second transformation
        xj = self.transform(sample)
        # Gaussian blur
        if random.random() > 0.5:
            sigma = 0.01 + (random.random() * 2)
            kornia_gauss = GaussianBlur2dKornia(kernel_size=(5, 5), sigma=(sigma, sigma))
            xj = kornia_gauss(xj.unsqueeze(0)).squeeze(0)

        return xi, xj


def _get_transforms(s, input_shape):
    r"""Returns compositions of transforms similar to what described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor()])
    return data_transforms


def get_coco_dataloaders(config):
    r"""Returns train and validation dataloaders."""
    data_transforms = _get_transforms(config["dataset"]["s"], config["dataset"]["input_shape"])
    train_dataset = COCO2014(config["dataset"]["dataset_path"], 
                             transform=SimCLRDualTransform(data_transforms),
                             mode="train")
    val_dataset =  COCO2014(config["dataset"]["dataset_path"], 
                            transform=SimCLRDualTransform(data_transforms),
                            mode="val")                           

    train_loader = DataLoader(train_dataset, 
                              batch_size=config["batch_size"],
                              num_workers=config["dataset"]["num_workers"], 
                              drop_last=True, 
                              shuffle=True)

    valid_loader = DataLoader(val_dataset, 
                              batch_size=config["batch_size"], 
                              num_workers=config["dataset"]["num_workers"], 
                              drop_last=True)

    return train_loader, valid_loader

