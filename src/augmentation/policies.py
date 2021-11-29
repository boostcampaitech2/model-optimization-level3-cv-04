"""PyTorch transforms for data augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import torchvision.transforms as transforms

from src.augmentation.methods import RandAugmentation, SequentialAugmentation
from src.augmentation.transforms import FILLCOLOR, SquarePad

DATASET_NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "TUNE": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}


def simple_augment_train(
    dataset: str = "CIFAR10", img_size: float = 32
) -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),
            transforms.RandomResizedCrop(
                size=img_size, ratio=(0.75, 1.0, 1.3333333333333333)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )


def simple_augment_test(
    dataset: str = "CIFAR10", img_size: float = 32
) -> transforms.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )


def randaugment_train(
    dataset: str = "CIFAR10",
    img_size: float = 32,
    n_select: int = 2,
    level: int = 14,
    n_level: int = 31,
) -> transforms.Compose:
    """Random augmentation policy for training CIFAR100."""
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 0.8, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )


def custom_augment_train(
    dataset: str = "CIFAR10", img_size: float = 32
) -> transforms.Compose:
    """Custom data augmentation rule for training TACO."""
    return transforms.Compose(
        [
            transforms.Resize((int(img_size), int(img_size))),
            transforms.RandomResizedCrop(
                size=img_size, ratio=(0.75, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def albu_heavy_train(
    dataset: str = "CIFAR10", img_size: float = 32
) -> A.Compose:
    """Custom data augmentation rule for training TACO."""
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([
            A.Flip(p=1.0),
            A.RandomRotate90(p=1.0)
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
        A.GaussNoise(p=0.3),
        A.OneOf([
            A.Blur(p=1.0), 
            A.GaussianBlur(p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0), 
            A.MotionBlur(p=1.0)
        ], p=0.1), 
        A.CLAHE(p=0.01),
        ToTensorV2()
    ])