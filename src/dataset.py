import torch.utils.data as data
from torchvision.datasets.vision import VisionDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import torchvision.datasets as datasets
from torchvision.datasets.folder import ImageFolder, make_dataset, default_loader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class AlbuImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=sample)
            sample = augmented['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target