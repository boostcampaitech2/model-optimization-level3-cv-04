"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import wandb

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
from src.utils.setseed import setSeed

from swin.models import build_model
from swin.config import get_config
from collections import OrderedDict

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)
    model_config = get_config(args.model)

    model = build_model(model_config)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")

    # load Pretrained model
    checkpoint = os.path.join(f"swin/saved/{model_config.MODEL.NAME}.pt")
    if os.path.isfile(checkpoint):
        print(f"Model load: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location=device)['model']
        # num_classes가 달라서 head의 weight는 불러오지 않음
        temp = OrderedDict()
        for n, v in state_dict.items():
            name = n.replace("head.","") 
            temp[name] = v
        model.load_state_dict(temp, strict=False)

    model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )
    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="swin/configs/swin_mlp_tiny_c24_patch4_window8_256.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco_swin.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--run_name", default="exp", type=str, help="run name for wandb"
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="seed"
    )
    parser.add_argument(
        "--save_dir", default="latest", type=str, help="exp name"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)
    setSeed(args.seed)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)
    
    # for wandb
    wandb.init(project='lightweight', entity='cv4', name = args.run_name, save_code = True)
    wandb.run.name = args.run_name
    wandb.run.save()
    wandb.config.update(model_config)
    wandb.config.update(data_config)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

