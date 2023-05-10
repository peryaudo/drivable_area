from glob import glob
import os
from typing import Any
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision.transforms import functional as F

class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

def get_transforms():
    return Compose([
        PILToTensor(),
        ConvertImageDtype(torch.float),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class DrivableAreaDataset(VisionDataset):
    def __init__(self, root, phase='train', transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)

        def get_image_path(label_path):
            base, ext = os.path.splitext(os.path.basename(label_path))
            return os.path.join(root, 'images', '100k', phase, base + '.jpg')

        self.label_paths = glob(root + '/labels/drivable/masks/' + phase + '/*')
        self.image_paths = [get_image_path(label_path) for label_path in self.label_paths]

    def __getitem__(self, index: int) -> Any:
        img = Image.open(self.image_paths[index]).convert('RGB')
        target = Image.open(self.label_paths[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.image_paths)
