"""Tune Model Architecture. 
- Author: SungJin Park
- Contact: 8639sung@gmail.com
"""
import optuna
import copy
import os
from datetime import datetime
import wandb
import torch
import torch.nn as nn
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime
from src.utils.common import read_yaml
from train import train
from src.trainer import count_model_params
from typing import Any, Dict, List, Tuple
import argparse

DATA_CONFIG = read_yaml(cfg="configs/data/taco_tune.yaml")
MODEL_CONFIG = read_yaml(cfg="configs/model/efficientnet_tune.yaml")

def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    n_stride = 0
    MAX_NUM_STRIDE = 5
    UPPER_STRIDE = 2  

    # Module 1
    m1 = trial.suggest_categorical("m1", ["Conv", "DWConv"])
    m1_args = []
    m1_repeat = trial.suggest_int("m1/repeat", 1, 3)
    m1_out_channel = trial.suggest_int("m1/out_channels", low=16, high=24, step=8)
    m1_stride = trial.suggest_int("m1/stride", low=1, high=UPPER_STRIDE)
    if m1_stride == 2:
        n_stride += 1
    m1_activation = trial.suggest_categorical("m1/activation", ["ReLU", "Hardswish"])
    if m1 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, 1, m1_activation]
    elif m1 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, m1_activation]
    model.append([m1_repeat, m1, m1_args])

    # Module 2
    m2 = trial.suggest_categorical(
        "m2", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "MBConv",
        "ShuffleNetV2", "Pass"]
    )
    m2_args = []
    m2_repeat = trial.suggest_int("m2/repeat", 1, 5)
    m2_out_channel = trial.suggest_int("m2/out_channels", low=16, high=128, step=16)
    m2_stride = trial.suggest_int("m2/stride", low=1, high=UPPER_STRIDE)
    # force stride m2
    if n_stride == 0:
        m2_stride = 2
    if m2 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m2_kernel = trial.suggest_int("m2/kernel_size", low=1, high=5, step=2)
        m2_activation = trial.suggest_categorical(
            "m2/activation", ["ReLU", "Hardswish"]
        )
        m2_args = [m2_out_channel, m2_kernel, m2_stride, None, 1, m2_activation]
    elif m2 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m2_kernel = trial.suggest_int("m2/kernel_size", low=1, high=5, step=2)
        m2_activation = trial.suggest_categorical(
            "m2/activation", ["ReLU", "Hardswish"]
        )
        m2_args = [m2_out_channel, m2_kernel, m2_stride, None, m2_activation]
    elif m2 == "InvertedResidualv2":
        m2_c = trial.suggest_int("m2/v2_c", low=16, high=32, step=16)
        m2_t = trial.suggest_int("m2/v2_t", low=1, high=4)
        m2_args = [m2_c, m2_t, m2_stride]
    elif m2 == "InvertedResidualv3":
        m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
        m2_t = round(trial.suggest_float("m2/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m2_c = trial.suggest_int("m2/v3_c", low=16, high=40, step=8)
        m2_se = trial.suggest_categorical("m2/v3_se", [0, 1])
        m2_hs = trial.suggest_categorical("m2/v3_hs", [0, 1])
        # k t c SE HS s
        m2_args = [m2_kernel, m2_t, m2_c, m2_se, m2_hs, m2_stride]
    elif m2 == "MBConv":
        m2_t = trial.suggest_int("m2/MB_t", low=1, high=3)
        m2_c = trial.suggest_int("m2/MB_c", low=16, high=40, step=8)
        m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
        m2_args = [m2_t, m2_c, m2_stride, m2_kernel]
    elif m2 == "ShuffleNetV2":
        m2_c = m1_out_channel * 2
        m2_args = [m2_stride]
    if not m2 == "Pass":
        if m2_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m2_repeat, m2, m2_args])

    # Module 3
    m3 = trial.suggest_categorical(
        "m3", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "MBConv",
        "ShuffleNetV2", "Pass"]
    )
    m3_args = []
    m3_repeat = trial.suggest_int("m3/repeat", 1, 5)
    m3_stride = trial.suggest_int("m3/stride", low=1, high=UPPER_STRIDE)
    if m3 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m3_out_channel = trial.suggest_int("m3/out_channels", low=16, high=128, step=16)
        m3_kernel = trial.suggest_int("m3/kernel_size", low=1, high=5, step=2)
        m3_activation = trial.suggest_categorical(
            "m3/activation", ["ReLU", "Hardswish"]
        )
        m3_args = [m3_out_channel, m3_kernel, m3_stride, None, 1, m3_activation]
    elif m3 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m3_out_channel = trial.suggest_int("m3/out_channels", low=16, high=128, step=16)
        m3_kernel = trial.suggest_int("m3/kernel_size", low=1, high=5, step=2)
        m3_activation = trial.suggest_categorical(
            "m3/activation", ["ReLU", "Hardswish"]
        )
        m3_args = [m3_out_channel, m3_kernel, m3_stride, None, m3_activation]
    elif m3 == "InvertedResidualv2":
        m3_c = trial.suggest_int("m3/v2_c", low=8, high=32, step=8)
        m3_t = trial.suggest_int("m3/v2_t", low=1, high=8)
        m3_args = [m3_c, m3_t, m3_stride]
    elif m3 == "InvertedResidualv3":
        m3_kernel = trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
        m3_t = round(trial.suggest_float("m3/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m3_c = trial.suggest_int("m3/v3_c", low=8, high=40, step=8)
        m3_se = trial.suggest_categorical("m3/v3_se", [0, 1])
        m3_hs = trial.suggest_categorical("m3/v3_hs", [0, 1])
        m3_args = [m3_kernel, m3_t, m3_c, m3_se, m3_hs, m3_stride]
    elif m3 == "MBConv":
        m3_t = trial.suggest_int("m3/MB_t", low=1, high=6)
        m3_c = trial.suggest_int("m3/MB_c", low=8, high=40, step=8)
        m3_kernel = trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
        m3_args = [m3_t, m3_c, m3_stride, m3_kernel]
    elif m3 == "ShuffleNetV2":
        m3_c = m2_out_channel
        m3_args = [m3_stride]
    if not m3 == "Pass":
        if m3_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m3_repeat, m3, m3_args])

    # Module 4
    m4 = trial.suggest_categorical(
        "m4", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "MBConv",
        "ShuffleNetV2", "Pass"]
    )
    m4_args = []
    m4_repeat = trial.suggest_int("m4/repeat", 1, 5)
    m4_stride = trial.suggest_int("m4/stride", low=1, high=UPPER_STRIDE)
    # force stride m4
    if n_stride == 1:
        m4_stride = 2
    if m4 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m4_out_channel = trial.suggest_int("m4/out_channels", low=16, high=256, step=16)
        m4_kernel = trial.suggest_int("m4/kernel_size", low=1, high=5, step=2)
        m4_activation = trial.suggest_categorical(
            "m4/activation", ["ReLU", "Hardswish"]
        )
        m4_args = [m4_out_channel, m4_kernel, m4_stride, None, 1, m4_activation]
    elif m4 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m4_out_channel = trial.suggest_int("m4/out_channels", low=16, high=256, step=16)
        m4_kernel = trial.suggest_int("m4/kernel_size", low=1, high=5, step=2)
        m4_activation = trial.suggest_categorical(
            "m4/activation", ["ReLU", "Hardswish"]
        )
        m4_args = [m4_out_channel, m4_kernel, m4_stride, None, m4_activation]
    elif m4 == "InvertedResidualv2":
        m4_c = trial.suggest_int("m4/v2_c", low=8, high=64, step=8)
        m4_t = trial.suggest_int("m4/v2_t", low=1, high=8)
        m4_args = [m4_c, m4_t, m4_stride]
    elif m4 == "InvertedResidualv3":
        m4_kernel = trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
        m4_t = round(trial.suggest_float("m4/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m4_c = trial.suggest_int("m4/v3_c", low=8, high=80, step=8)
        m4_se = trial.suggest_categorical("m4/v3_se", [0, 1])
        m4_hs = trial.suggest_categorical("m4/v3_hs", [0, 1])
        m4_args = [m4_kernel, m4_t, m4_c, m4_se, m4_hs, m4_stride]
    elif m4 == "MBConv":
        m4_t = trial.suggest_int("m4/MB_t", low=1, high=6)
        m4_c = trial.suggest_int("m4/MB_c", low=8, high=80, step=8)
        m4_kernel = trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
        m4_args = [m4_t, m4_c, m4_stride, m4_kernel]
    elif m4 == "ShuffleNetV2":
        m4_c = m3_out_channel * 2
        m4_args = [m4_stride]
    if not m4 == "Pass":
        if m4_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m4_repeat, m4, m4_args])

    # Module 5
    m5 = trial.suggest_categorical(
        "m5", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "MBConv",
        "ShuffleNetV2", "Pass"]
    )
    m5_args = []
    m5_repeat = trial.suggest_int("m5/repeat", 1, 5)
    m5_stride = 1
    if m5 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m5_out_channel = trial.suggest_int("m5/out_channels", low=16, high=256, step=16)
        m5_kernel = trial.suggest_int("m5/kernel_size", low=1, high=5, step=2)
        m5_activation = trial.suggest_categorical(
            "m5/activation", ["ReLU", "Hardswish"]
        )
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_out_channel, m5_kernel, m5_stride, None, 1, m5_activation]
    elif m5 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m5_out_channel = trial.suggest_int("m5/out_channels", low=16, high=256, step=16)
        m5_kernel = trial.suggest_int("m5/kernel_size", low=1, high=5, step=2)
        m5_activation = trial.suggest_categorical(
            "m5/activation", ["ReLU", "Hardswish"]
        )
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_out_channel, m5_kernel, m5_stride, None, m5_activation]
    elif m5 == "InvertedResidualv2":
        m5_c = trial.suggest_int("m5/v2_c", low=16, high=128, step=16)
        m5_t = trial.suggest_int("m5/v2_t", low=1, high=8)
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_c, m5_t, m5_stride]
    elif m5 == "InvertedResidualv3":
        m5_kernel = trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
        m5_t = round(trial.suggest_float("m5/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m5_c = trial.suggest_int("m5/v3_c", low=16, high=80, step=16)
        m5_se = trial.suggest_categorical("m5/v3_se", [0, 1])
        m5_hs = trial.suggest_categorical("m5/v3_hs", [0, 1])
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_kernel, m5_t, m5_c, m5_se, m5_hs, m5_stride]
    elif m5 == "MBConv":
        m5_t = trial.suggest_int("m5/MB_t", low=1, high=6)
        m5_c = trial.suggest_int("m5/MB_c", low=16, high=80, step=16)
        m5_kernel = trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
        m5_args = [m5_t, m5_c, m5_stride, m5_kernel]
    elif m5 == "ShuffleNetV2":
        m5_c = m4_out_channel
        m5_args = [m5_stride]
    if not m5 == "Pass":
        if m5_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m5_repeat, m5, m5_args])

    # Module 6
    m6 = trial.suggest_categorical(
        "m6", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "MBConv",
        "ShuffleNetV2", "Pass"]
    )
    m6_args = []
    m6_repeat = trial.suggest_int("m6/repeat", 1, 5)
    m6_stride = trial.suggest_int("m6/stride", low=1, high=UPPER_STRIDE)
    # force stride m6
    if n_stride == 2:
        m4_stride = 2
    if m6 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m6_out_channel = trial.suggest_int("m6/out_channels", low=16, high=512, step=16)
        m6_kernel = trial.suggest_int("m6/kernel_size", low=1, high=5, step=2)
        m6_activation = trial.suggest_categorical(
            "m6/activation", ["ReLU", "Hardswish"]
        )
        m6_args = [m6_out_channel, m6_kernel, m6_stride, None, 1, m6_activation]
    elif m6 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m6_out_channel = trial.suggest_int("m6/out_channels", low=16, high=512, step=16)
        m6_kernel = trial.suggest_int("m6/kernel_size", low=1, high=5, step=2)
        m6_activation = trial.suggest_categorical(
            "m6/activation", ["ReLU", "Hardswish"]
        )
        m6_args = [m6_out_channel, m6_kernel, m6_stride, None, m6_activation]
    elif m6 == "InvertedResidualv2":
        m6_c = trial.suggest_int("m6/v2_c", low=16, high=128, step=16)
        m6_t = trial.suggest_int("m6/v2_t", low=1, high=8)
        m6_args = [m6_c, m6_t, m6_stride]
    elif m6 == "InvertedResidualv3":
        m6_kernel = trial.suggest_int("m6/kernel_size", low=3, high=5, step=2)
        m6_t = round(trial.suggest_float("m6/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m6_c = trial.suggest_int("m6/v3_c", low=16, high=160, step=16)
        m6_se = trial.suggest_categorical("m6/v3_se", [0, 1])
        m6_hs = trial.suggest_categorical("m6/v3_hs", [0, 1])
        m6_args = [m6_kernel, m6_t, m6_c, m6_se, m6_hs, m6_stride]
    elif m6 == "MBConv":
        m6_t = trial.suggest_int("m6/MB_t", low=1, high=6)
        m6_c = trial.suggest_int("m6/MB_c", low=16, high=160, step=16)
        m6_kernel = trial.suggest_int("m6/kernel_size", low=3, high=5, step=2)
        m6_args = [m6_t, m6_c, m6_stride, m6_kernel]
    elif m6 == "ShuffleNetV2":
        m6_c = m5_out_channel * 2
        m6_args = [m6_stride]
    if not m6 == "Pass":
        if m6_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m6_repeat, m6, m6_args])

    # Module 7
    m7 = trial.suggest_categorical(
        "m7", ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "MBConv",
        "ShuffleNetV2", "Pass"]
    )
    m7_args = []
    m7_repeat = trial.suggest_int("m7/repeat", 1, 5)
    m7_stride = trial.suggest_int("m7/stride", low=1, high=UPPER_STRIDE)
    if m7 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m7_out_channel = trial.suggest_int(
            "m7/out_channels", low=128, high=1024, step=128
        )
        m7_kernel = trial.suggest_int("m7/kernel_size", low=1, high=5, step=2)
        m7_activation = trial.suggest_categorical(
            "m7/activation", ["ReLU", "Hardswish"]
        )
        m7_args = [m7_out_channel, m7_kernel, m7_stride, None, 1, m7_activation]
    elif m7 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m7_out_channel = trial.suggest_int(
            "m7/out_channels", low=128, high=1024, step=128
        )
        m7_kernel = trial.suggest_int("m7/kernel_size", low=1, high=5, step=2)
        m7_activation = trial.suggest_categorical(
            "m7/activation", ["ReLU", "Hardswish"]
        )
        m7_args = [m7_out_channel, m7_kernel, m7_stride, None, m7_activation]
    elif m7 == "InvertedResidualv2":
        m7_c = trial.suggest_int("m7/v2_c", low=16, high=160, step=16)
        m7_t = trial.suggest_int("m7/v2_t", low=1, high=8)
        m7_args = [m7_c, m7_t, m7_stride]
    elif m7 == "InvertedResidualv3":
        m7_kernel = trial.suggest_int("m7/kernel_size", low=3, high=5, step=2)
        m7_t = round(trial.suggest_float("m7/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m7_c = trial.suggest_int("m7/v3_c", low=8, high=160, step=8)
        m7_se = trial.suggest_categorical("m7/v3_se", [0, 1])
        m7_hs = trial.suggest_categorical("m7/v3_hs", [0, 1])
        m7_args = [m7_kernel, m7_t, m7_c, m7_se, m7_hs, m7_stride]
    elif m7 == "MBConv":
        m7_t = trial.suggest_int("m7/MB_t", low=1, high=6)
        m7_c = trial.suggest_int("m7/MB_c", low=8, high=160, step=8)
        m7_kernel = trial.suggest_int("m7/kernel_size", low=3, high=5, step=2)
        m7_args = [m7_t, m7_c, m7_stride, m7_kernel]
    elif m7 == "ShuffleNetV2":
        m7_c = m6_out_channel
        m7_args = [m7_stride]
    if not m7 == "Pass":
        if m7_stride == 2:
            n_stride += 1
            if n_stride >= MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m7_repeat, m7, m7_args])

    # last layer
    last_dim = trial.suggest_int("last_dim", low=128, high=1024, step=128)
    # We can setup fixed structure as well
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "FixedConv", [6, 1, 1, None, 1, None]])

    module_info = {}
    module_info["m1"] = {"type": m1, "repeat": m1_repeat, "stride": m1_stride}
    module_info["m2"] = {"type": m2, "repeat": m2_repeat, "stride": m2_stride}
    module_info["m3"] = {"type": m3, "repeat": m3_repeat, "stride": m3_stride}
    module_info["m4"] = {"type": m4, "repeat": m4_repeat, "stride": m4_stride}
    module_info["m5"] = {"type": m5, "repeat": m5_repeat, "stride": m5_stride}
    module_info["m6"] = {"type": m6, "repeat": m6_repeat, "stride": m6_stride}
    module_info["m7"] = {"type": m7, "repeat": m7_repeat, "stride": m7_stride}

    return model, module_info


def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    model_config = copy.deepcopy(MODEL_CONFIG)
    data_config = copy.deepcopy(DATA_CONFIG)
    
    # save dir
    log_dir = os.path.join("exp/NAS", datetime.now().strftime(f"Trial_{trial.number}_%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # wandb init
    wandb.init(entity="cv4",
                project='lightweight',
                group="NAS",
                name=f'Trial_{trial.number}',
                config=model_config,
                reinit=True
                )

    # model config
    model_config["INPUT_SIZE"] = [data_config["IMG_SIZE"], data_config["IMG_SIZE"]]
    model_config["backbone"], module_info = search_model(trial)
 
    model = Model(model_config, verbose=True)
    model.to(device)
    model.model.to(device)

    model_info(model, verbose=True)

    # check current model runtime
    mean_time = check_runtime(
        model.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )

    # check current model params_nums
    params_nums = count_model_params(model)

    # skip unsuitable model 
    if mean_time >= 10: 
        print(f' trial: {trial.number}, This model takes too much time:{mean_time}')
        raise optuna.structs.TrialPruned()

    if params_nums >= 1000000:
        print(f' trial: {trial.number}, This model has too many param:{params_nums}')
        raise optuna.structs.TrialPruned()

    # train current model
    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
    
    wandb.log({'f1':test_f1,'params_nums':params_nums, 'mean_time':mean_time})
    
    return test_f1, params_nums, mean_time


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, seed, storage: str = None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler(seed=seed)
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="NAS",
        sampler=sampler,
        storage=rdb_storage, 
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, device), n_trials=200)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="", type=str, help="Optuna database storage path.")
    parser.add_argument("--seed", default=42, type=int, help="Sampler seed")
    args = parser.parse_args()
    tune(args.gpu, args.seed, storage=args.storage if args.storage != "" else None)