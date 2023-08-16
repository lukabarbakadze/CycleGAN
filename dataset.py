import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import pytorch_lightning as pl



class MainDataset(Dataset):
    def __init__(self, root_monet, root_other, transform):
        self.root_monet = root_monet
        self.root_other = root_other
        self.transform = transform

        self.monet_imgs = os.listdir(root_monet)
        self.other_imgs = os.listdir(root_other)

        self.monet_len = len(self.monet_imgs)
        self.other_len = len(self.other_imgs)

        self.length_dataset = max(self.monet_len, self.other_len)


    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        monet_img = self.monet_imgs[idx % self.monet_len]
        other_img = self.other_imgs[idx % self.other_len]

        monet_path = os.path.join(self.root_monet, monet_img)
        other_path = os.path.join(self.root_other, other_img)

        # monet_img = np.transpose(np.array(Image.open(monet_path).convert("RGB")), (2, 0, 1))
        # other_img = np.transpose(np.array(Image.open(other_path).convert("RGB")), (2, 0, 1))
        monet_img = Image.open(monet_path).convert("RGB")
        other_img = Image.open(other_path).convert("RGB")
        
        monet_img = self.transform(monet_img)
        other_img = self.transform(other_img)

        return monet_img, other_img

class CycleGanDataset(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, transform):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage):
        # done on multiple GPU
        entire_dataset = MainDataset(
            root_monet = os.path.join(self.data_dir, "monet_jpg"),
            root_other = os.path.join(self.data_dir, "photo_jpg"),
            transform = self.transform
        )
        self.train_ds = entire_dataset
        # self.train_ds, self.val_ds = random_split(entire_dataset, [0.9, 0.1])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_ds,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False
    #     )
    
# class ImageTransform:
#     def __init__(self, img_size=256):
#         self.transform = transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor()
#         ])

#     def __call__(self, img):
#         img = self.transform(img)
#         return img