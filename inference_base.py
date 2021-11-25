"""Example code for submit.

- Author: Junghoon Kim, Jongkuk Lim
- Contact: placidus36@gmail.com, lim.jeikei@gmail.com
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize
from tqdm import tqdm

from src.augmentation.policies import simple_augment_test
from src.model import Model
from src.utils.common import read_yaml

if torch.__version__ >= "1.8.1":
    from torch import profiler
else:
    from torch.autograd import profiler

CLASSES = [
    "Metal",
    "Paper",
    "Paperpack",
    "Plastic",
    "Plasticbag",
    "Styrofoam",
]


class CustomImageFolder(ImageFolder):
    """ImageFolder with filename."""

    def __getitem__(self, index):
        img_gt = super(CustomImageFolder, self).__getitem__(index)
        fdir = self.imgs[index][0]
        fname = fdir.rsplit(os.path.sep, 1)[-1]
        return img_gt + (fname,)


def get_dataloader(img_root: str, data_config: str) -> DataLoader:
    """Get dataloader.

    Note:
        Don't forget to set normalization.
    """
    # Load yaml
    data_config = read_yaml(data_config)

    transform_test_args = (
        data_confg["AUG_TEST_PARAMS"] if data_config.get("AUG_TEST_PARAMS") else None
    )
    # Transformation for test
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TEST"],
    )(dataset=data_config["DATASET"], img_size=data_config["IMG_SIZE"])

    dataset = CustomImageFolder(root=img_root, transform=transform_test)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=8)
    return dataloader


@torch.no_grad()
def inference(model, dataloader, dst_path: str, t0: float) -> None:
    """Run inference with given model and dataloader.

    Args:
        model: PyTorch model.
        dataloader: PyTorch dataset loader.
        dst_path: destination path for inference result to be written.
        t0: initial time prior to creating model and dataset
            by time.monotonic().
    """
    model = model.to(device)
    model.eval()

    profile_ = torch.rand(1, 3, 512, 512).to(device)
    for transform in dataloader.dataset.transform.transforms:
        if isinstance(transform, Resize):
            profile_input = torch.rand(1, 3, *transform.size).to(device)
            break

    n_profile = 100
    print(f"Profile input shape: {profile_input.shape}")
    with profiler.profile(use_cuda=True, profile_memory=False) as prof:
        for _ in tqdm(range(100), "Running profile ..."):
            x = model(profile_input)
    avg_time = prof.total_average()

    if hasattr(avg_time, "self_cuda_time_total"):
        cuda_time = avg_time.self_cuda_time_total / 1e6 / n_profile
    else:
        cuda_time = avg_time.cuda_time_total / 1e6 / n_profile

    cpu_time = avg_time.self_cpu_time_total / 1e6 / n_profile
    print(prof.key_averages())
    print(f"Average CUDA time: {cuda_time}, CPU time: {cpu_time}")

    result = {
        "inference": {},
        "time": {
            "profile": {"cuda": float("inf"), "cpu": float("inf")},
            "runtime": {"all": 0, "inference_only": 0},
            "inference": {},
        },
        "macs": float("inf"),
    }
    time_measure_inference = 0
    for img, _, fname in tqdm(dataloader, "Running inference ..."):
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)

        t_start.record()
        img = img.to(device)
        pred = model(img)
        pred = torch.argmax(pred)

        t_end.record()
        torch.cuda.synchronize()
        t_inference = t_start.elapsed_time(t_end) / 1000
        time_measure_inference += t_inference

        result["inference"][fname[0]] = CLASSES[int(pred.detach())]
        result["time"]["inference"][fname[0]] = t_inference

    result["time"]["profile"]["cuda"] = cuda_time
    result["time"]["profile"]["cpu"] = cpu_time
    result["time"]["runtime"]["all"] = time.monotonic() - t0
    result["time"]["runtime"]["inference_only"] = time_measure_inference

    j = json.dumps(result, indent=4)
    save_path = os.path.join(dst_path, "output.csv")
    with open(save_path, "w") as outfile:
        json.dump(result, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit.")
    parser.add_argument(
        "--dst", type=str, help="destination path for submit",
        default=os.environ.get('SM_OUTPUT_DATA_DIR')
    )
    parser.add_argument("--model_dir", type=str, help="Saved model root directory which includes 'best.pt', 'data.yml', and, 'model.yml'", default='/opt/ml/code/exp/latest')
    parser.add_argument("--weight_name", type=str, help="Model weight file name. (best.pt, best.ts, ...)", default="best.pt")
    parser.add_argument(
        "--img_root",
        type=str,
        help="image folder root. e.g) 'data/test'",
        default='/opt/ml/data/test'
    )
    args = parser.parse_args()
    assert args.model_dir != '' and args.img_root != '', "'--model_dir' and '--img_root' must be provided."

    args.weight = os.path.join(args.model_dir, args.weight_name)
    args.model_config = os.path.join(args.model_dir, "model.yml")
    args.data_config = os.path.join(args.model_dir, "data.yml")

    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare datalaoder
    dataloader = get_dataloader(img_root=args.img_root, data_config=args.data_config)

    # prepare model
    if args.weight.endswith("ts"):
        model = torch.jit.load(args.weight)
    else:
        model_instance = Model(args.model_config, verbose=True)
        model_instance.model.load_state_dict(
            torch.load(args.weight, map_location=torch.device("cpu"))
        )
        model = model_instance.model

    # inference
    inference(model, dataloader, args.dst, t0)

