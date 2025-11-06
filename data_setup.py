import os
from pathlib import Path
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

NUM_WORKERS = os.cpu_count()

class AganDataset(Dataset):
    def __init__(self, targ_dir: str, transform=None, seed=42):
        # first = Path(targ_dir)
        # next_one = first.glob("data/*")
        # next_two = list(next_one)
        data_ext = list(Path(targ_dir).glob("data/*"))[0].suffix
        gt_ext = list(Path(targ_dir).glob("gt/*"))[0].suffix
        
        self.data_paths = sorted(list(Path(targ_dir).glob(f"data/*{data_ext}")))
        self.gt_paths = sorted(list(Path(targ_dir).glob(f"gt/*{gt_ext}")))
        self.seed = seed
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        data_path, gt_path = self.data_paths[index], self.gt_paths[index]
        return Image.open(data_path).convert("RGB"), Image.open(gt_path).convert("RGB")

    def get_length(self) -> Tuple[int, int]:
        return len(self.data_paths), len(self.gt_paths)
    
    def __len__(self) -> int:
        return min(len(self.data_paths), len(self.gt_paths))

    def __getitem__(self, index: int):
        data_img, gt_img = self.load_image(index)

        if self.transform:
            torch.manual_seed(self.seed)
            data_img = self.transform(data_img)
            torch.manual_seed(self.seed)
            gt_img = self.transform(gt_img)

        return data_img, gt_img


def create_dataloader(train_dir: str,
                      test_dir: str,
                      train_batch_size: int,
                      test_batch_size: int,
                      num_workers: int = NUM_WORKERS):

    # transform train and test data
    train_transform = transforms.Compose([
        transforms.CenterCrop(size=(480, 480)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(size=(480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Instantiate Dataset and DataLoader
    train_dataset = AganDataset(targ_dir = train_dir,
                                transform = train_transform)
    test_dataset = AganDataset(targ_dir = test_dir,
                               transform = test_transform)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)

    return train_dataloader, test_dataloader
