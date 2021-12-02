from re import L
from numpy.lib.function_base import kaiser
from src.utils.setseed import setSeed
from torchvision.transforms.transforms import Lambda
import wandb
import optuna
import joblib
import copy
import inspect
import argparse
from datetime import datetime
import os
import numpy as np
from typing import Any, Dict, Tuple, Union, List
import pickle
# torch
import torch
import torch.nn as nn

# module/class import
import src
import src.modules
from src.model import Model
from src.utils.common import read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import save_model
from src.trainer import TorchTrainer
from train import train
from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.setseed import setSeed

# decompose tensorly
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from tensorly import partial_svd

# warning 무시
import warnings 
warnings.filterwarnings('ignore')

"""
module_list : src내 class들을 list로 받는 역할
"""
MODULE_LIST = []
for name, obj in inspect.getmembers(src.modules): 
    if inspect.isclass(obj):
        MODULE_LIST.append(obj)


total_mse_score = 0
len_total_mse_score = 0
mse_config = {}


def get_mse(src, core, tucker_factors = None, option = 'tucker') -> float:
    """ Calc mse for decompose
        src  : pretrain된 conv weight
        tgt : decomposition을 통해 만들어진 conv weight
    """
    global total_mse_score
    global len_total_mse_score
    if option == 'tucker':
        tgt = tl.tucker_to_tensor((core, tucker_factors))
        if isinstance(src, torch.Tensor):
            mse_per_layer = torch.mean((src - tgt)**2)
            total_mse_score += mse_per_layer
            len_total_mse_score +=1
        elif isinstance(src, np.ndarray):
            mse_per_layer = np.mean((src - tgt)**2)
            total_mse_score += mse_per_layer
            len_total_mse_score +=1
    else :
        mse_per_layer = torch.mean((src - core)**2)
        total_mse_score += mse_per_layer
        len_total_mse_score +=1
        
    return mse_per_layer


class group_decomposition_conv(nn.Module):
    '''
    group 수에 맞춰 소분할 한 후, tucker_decomposition_conv_layer 함수를 거친 x값을 다시 concat 후 return 해줍니다. 
    ex) conv(24,36,kernel_size = 3 , groups = 4)
        -> conv(6,9,kernel_size = 3 , groups = 1)을 decompose하는 과정을총 4번 반복 후 concat
    '''
    def __init__(self, layer : nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.n_groups = layer.groups
        self.in_channel = int(layer.in_channels / self.n_groups)
        self.out_channel = int(layer.out_channels / self.n_groups)

        self.conv_module = nn.Conv2d(self.in_channel , self.out_channel , kernel_size = layer.kernel_size , stride = layer.stride , groups = 1, padding = layer.padding, bias = False)
        self.conv_list = []
        for i in range(self.n_groups):
            self.conv_module.weight.data  = layer.weight.data[self.out_channel * i : self.out_channel *(i+1)]
            self.conv_list.append(self.conv_module)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.chunk(self.n_groups , dim = 1)  
        decompose_x = []
        for value, temp_conv in zip(xs, self.conv_list):
            decompose_x.append(tucker_decomposition_conv_layer(temp_conv)(value))
        out = torch.cat(decompose_x , dim = 1)
        return out


def pointwise_decomposition_conv_layer(
    layer : nn.Module, 
    normed_rank : int = 0.4
    ) -> nn.Module: 
    '''
        pointwiseconv에 대해서만 SVD decompose 실행
        (o, i, 1, 1) 를 (o, i) 2차원으로 view 해준 후 svd 실행
        rank를 따로 받지 않고 i, o 상관없이 비율로 고정해서 진행
    '''
    i = layer.in_channels   
    o = layer.out_channels  
    rank = min(int(o * normed_rank), i) 
    u , s , v = partial_svd(layer.weight.data.view(o, i) , n_eigenvecs = rank) 
    u = torch.matmul(u,torch.diag(s))
    tgt = torch.matmul(u, v)
    mse_per_layer = get_mse(layer.weight.data , tgt.unsqueeze(2).unsqueeze(3), option = 'svd')
    u = u.view(o , rank , 1, 1)
    v = v.view(rank , i ,  1, 1)
    
    U_conv = nn.Conv2d(rank , o , 1, 1, 0, bias = False)
    V_conv = nn.Conv2d(i , rank , 1, 1, 0, bias=True if hasattr(layer, "bias") and layer.bias is not None else False)
    U_conv.weight.data = u
    V_conv.weight.data = v

    new_layers = [V_conv , U_conv]

    global mse_config
    mse_config[layer]  = mse_per_layer
    return nn.Sequential(*new_layers)  


def depthwise_decomposition_conv_layer( 
    layer : nn.Module,  #
    normed_rank : int = 1
    ) -> nn.Module :
    '''
        depthwise에 대해서만 SVD decompose 실행
        (o, i, k, k)를 (k,k) 2차원으로 view해준 후, svd를 o만큼 실행
    '''
    if layer.stride[0] != 1 :
        return layer

    k = layer.kernel_size[0]
    new_u = torch.zeros(layer.weight.shape[0],1,k,1)
    new_v = torch.zeros(layer.weight.shape[0],1,1,k)
    for o in range(layer.weight.shape[0]):
        u , s , v = partial_svd(layer.weight.data[o].view(k, k) , n_eigenvecs = normed_rank)
        u *= torch.sqrt(s)
        v *= torch.sqrt(s)
        new_u[o] = u.view(1,k,normed_rank)
        new_v[o] = v.view(1,normed_rank,k)

    U_conv = nn.Conv2d(layer.weight.shape[0],
                        layer.weight.shape[0],
                        kernel_size = (k , normed_rank),
                        stride = (layer.stride[0] , layer.stride[1]),
                        groups = layer.groups,
                        padding = (layer.padding[0],0),
                        bias = False)

    V_conv = nn.Conv2d(layer.weight.shape[0],
                        layer.weight.shape[0],
                        kernel_size = (normed_rank , k),
                        stride = (1,1),
                        groups = layer.groups,
                        padding = (0,layer.padding[1]),
                        bias=True if hasattr(layer, "bias") and layer.bias is not None else False)
    
    U_conv.weight.data = new_u
    V_conv.weight.data = new_v

    new_layers = [U_conv , V_conv]
    tgt = torch.matmul(new_u , new_v)
    mse_per_layer = get_mse(layer.weight.data , tgt, option = 'svd')
    
    global mse_config
    mse_config[layer]  = mse_per_layer
    return nn.Sequential(*new_layers)


def tucker_decomposition_conv_layer(
      layer: nn.Module,
      normed_rank: List[int] = [0.5, 0.5],
    ) -> nn.Module:
    """
    일반 conv / dw가 아닌 group conv에 대해서만 tucker decopmosition 수행 
    rank를 받아서 그 rank에 받게 decompositoin한 conv layer들을 sequential 형태로 return 해줌
    """
    if layer.in_channels == 1 or layer.out_channels == 1 :
        return layer
    if layer.groups != 1 :
        return layer

    if hasattr(layer, "rank"):
        normed_rank = getattr(layer, "rank")
    rank = [int(r * layer.weight.shape[i]) for i, r in enumerate(normed_rank)] 
    rank = [max(r, 2) for r in rank]


    core, [last, first] = partial_tucker(
        layer.weight.data,
        modes=[0, 1],
        n_iter_max=2000000,
        rank=rank,
        init="svd",
    )
 
    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )
    
    core_layer = nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    last_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True if hasattr(layer, "bias") and layer.bias is not None else False,
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = (
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    )
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    mse_per_layer = get_mse(layer.weight.data, core, [last, first] )
    global mse_config
    mse_config[layer]  = mse_per_layer

    print(*new_layers)
    return nn.Sequential(*new_layers)


def module_decompose(model_layers : nn.Module):
    """
    각 모듈 별 위치에 따른 decompose 
    """
    if type(model_layers) == src.modules.conv.Conv :
        if model_layers.conv.kernel_size == (1 , 1) :
             model_layers.conv = pointwise_decomposition_conv_layer(model_layers.conv)
        else :
            if model_layers.conv.groups != 1 and model_layers.conv.in_channels/model_layers.conv.groups != 1 and  model_layers.conv.out_channels/model_layers.conv.groups != 1 :
                model_layers.conv = group_decomposition_conv(model_layers.conv)
            else : 
                model_layers.conv = tucker_decomposition_conv_layer(model_layers.conv)

    elif type(model_layers) ==src.modules.dwconv.DWConv:
        if model_layers.conv.groups != 1 and model_layers.conv.in_channels/model_layers.conv.groups != 1 and  model_layers.conv.out_channels/model_layers.conv.groups != 1 :
            model_layers.conv = group_decomposition_conv(model_layers.conv)
     
    elif type(model_layers) == src.modules.mbconv.MBConv:
        if len(model_layers.conv) == 4 :
            model_layers.conv[0][1] = depthwise_decomposition_conv_layer(model_layers.conv[0][1]) 
            model_layers.conv[1].se[1] = pointwise_decomposition_conv_layer(model_layers.conv[1].se[1] , )
            model_layers.conv[1].se[3] = pointwise_decomposition_conv_layer(model_layers.conv[1].se[3])
            model_layers.conv[2] = pointwise_decomposition_conv_layer(model_layers.conv[2])
        else :
            model_layers.conv[1][1] = depthwise_decomposition_conv_layer(model_layers.conv[1][1]) 
            model_layers.conv[2].se[1] = pointwise_decomposition_conv_layer(model_layers.conv[2].se[1])
            model_layers.conv[2].se[3] = pointwise_decomposition_conv_layer(model_layers.conv[2].se[3])
            model_layers.conv[3] = pointwise_decomposition_conv_layer(model_layers.conv[3])

    elif type(model_layers) == src.modules.invertedresidualv3.InvertedResidualv3 :
        if len(model_layers.conv) == 6 :
            model_layers.conv[0] = depthwise_decomposition_conv_layer(model_layers.conv[4]) 
            model_layers.conv[4] = pointwise_decomposition_conv_layer(model_layers.conv[4])
        else :
            if type(model_layers.conv[5]) == 'SqueezeExcitation':
                model_layers.conv[0] = pointwise_decomposition_conv_layer(model_layers.conv[0])
                model_layers.conv[3] = depthwise_decomposition_conv_layer(model_layers.conv[3])
                model_layers.conv[5].fc1 = pointwise_decomposition_conv_layer(model_layers.conv[5].fc1)
                model_layers.conv[5].fc2 = pointwise_decomposition_conv_layer(model_layers.conv[5].fc2)
                model_layers.conv[7] = pointwise_decomposition_conv_layer(model_layers.conv[7])
            else :
                model_layers.conv[0] = pointwise_decomposition_conv_layer(model_layers.conv[0])
                model_layers.conv[3] = depthwise_decomposition_conv_layer(model_layers.conv[3])  
                model_layers.conv[7] = pointwise_decomposition_conv_layer(model_layers.conv[7])  

    elif type(model_layers) == src.modules.invertedresidualv2.InvertedResidualv2 :
        if len(model_layers.conv) == 3 :
            model_layers.conv[0][0] = depthwise_decomposition_conv_layer(model_layers.conv[0][0]) 
            model_layers.conv[1] = pointwise_decomposition_conv_layer(model_layers.conv[1])
        else :
            model_layers.conv[0][0] = pointwise_decomposition_conv_layer(model_layers.conv[0][0]) 
            model_layers.conv[1][0] = depthwise_decomposition_conv_layer(model_layers.conv[1][0]) 
            model_layers.conv[2] = pointwise_decomposition_conv_layer(model_layers.conv[2])

    return model_layers


def decompose(module: nn.Module):
    """model을 받아서 각 module마다 decompose 해줌"""
    model_layers = list(module.children())
    all_new_layers = []
    for i in range(len(model_layers)):
        if type(model_layers[i]) in MODULE_LIST :
            all_new_layers.append(module_decompose(model_layers[i]))
        
        elif type(model_layers[i]) == nn.Sequential:
            temp = []
            for j in range(len(model_layers[i])):
                temp.append(module_decompose(model_layers[i][j]))
            all_new_layers.append(nn.Sequential(*temp))
        
        else :
            all_new_layers.append(model_layers[i])
    
    return nn.Sequential(*all_new_layers)


class Objective:
    def __init__(self, model_instance, data_config , run_name):     
        
        self.model_instance = model_instance # 학습된 원 모델
        self.idx_list = []
        self.idx_layer = 0
        self.data_config = data_config
        self.config = {}
        self.run_name = run_name

        macs = calc_macs(self.model_instance.model, (3, self.data_config["IMG_SIZE"], int(self.data_config["IMG_SIZE"]*0.723)))
        print(f"before decomposition macs: {macs}")  
    
    def __call__(self,trial):

        wandb.init(project='lightweight', 
                    entity='cv4',
                    name=f'No_Trial_{trial.number}',
                    group=self.run_name,
                    config=self.config,
                    reinit=True)

        ## init for get_mse
        global total_mse_score
        global len_total_mse_score
        global mse_config
        total_mse_score = 0
        len_total_mse_score = 0       

        ## Rank setting
        decompose_model =copy.deepcopy(self.model_instance)
        rank1, rank2 = self.search_rank(trial) 
        
        for name, param in decompose_model.model.named_modules():
            layer_num = name.split('.')[0]
            if layer_num.isnumeric() and layer_num not in self.idx_list:
                self.idx_layer = layer_num
                self.idx_list.append(self.idx_layer)
                # rank1, rank2=self.search_rank(trial)
            if isinstance(param, nn.Conv2d):
                param.register_buffer('rank', torch.Tensor([rank1, rank2]))
            
       ## Decompose model         
        decompose_model.model = decompose(decompose_model.model)
        decompose_model.model.to(device)
        print('---------decompose model check ------------')
        print(decompose_model.model)

        ## Calculate MACs
        macs = calc_macs(decompose_model.model, (3, self.data_config["IMG_SIZE"], int(self.data_config["IMG_SIZE"]*0.723)))
        print(f'**************** macs : {macs} ****************')
        
        ## Cal MSE err
        mse_err = total_mse_score / len_total_mse_score
        print(f'mse: {mse_err}')

        ########################## Validation for F1 score ###########################
        setSeed(42)
        train_dl, val_dl, test_dl = create_dataloader(self.data_config)

        criterion = CustomCriterion(
            samples_per_cls=get_label_counts(self.data_config["DATA_PATH"]+'/train')
            if self.data_config["DATASET"] == "TACO"
            else None,
            device=device,
            fp16 = self.data_config["FP16"],
            loss_type= "softmax", 
        )
        ## Create optimizer, scheduler, criterion
        optimizer = torch.optim.AdamW(
            decompose_model.model.parameters(), lr=self.data_config["INIT_LR"]
        )
        ## Amp loss scaler
        scaler = (
            torch.cuda.amp.GradScaler() if self.data_config["FP16"] and device != torch.device("cpu") else None
        )
        trainer = TorchTrainer(
            model=decompose_model.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            scaler=scaler,
            device=device,
            model_path=None,
            verbose=1,
        )
        val_loss, val_f1, val_acc = trainer.test(model=decompose_model.model , test_dataloader=val_dl)
        print('Done Validation !')
        ##############################################################################


        ## for wandb
        wandb.log({"mse_err": mse_err, 
                    "macs": macs , 
                    "loss" : val_loss,
                    "f1" : val_f1,
                    "acc" : val_acc}, step=trial.number)

        print('----------------------------------------------------------------------------------------------------')
        print(f'macs : {macs} || mse_err : {mse_err} || f1 : {val_f1} || loss : {val_loss} || acc : {val_acc}')
        print('----------------------------------------------------------------------------------------------------')

        ## save rank config.pkl 
        PATH = '/opt/ml/code/exp_rank'
        if not os.path.isdir(PATH):
            os.mkdir(PATH)
        SAVE_PATH = os.path.join(PATH , self.run_name + f'_{trial.number}.pkl')
        with open(SAVE_PATH , 'wb') as fw:
            pickle.dump(self.config , fw)
            print(f'Save the config : {SAVE_PATH}')

        return val_f1, macs
    
    def search_rank(self,trial):
        para_name1 = f'{self.idx_layer}_layer_rank1'
        para_name2 =f'{self.idx_layer}_layer_rank2'
        rank1 = trial.suggest_float(para_name1, low = 0.1, high = 0.5, step = 0.05)
        rank2 = trial.suggest_float(para_name2 , low = 0.1, high = 0.5, step = 0.05)        
        self.config[para_name1]=rank1   
        self.config[para_name2]=rank2      
        return rank1, rank2


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="findRank.")
    parser.add_argument(
        "--weight", default="exp/79_0.1_10/best.pt", type=str, help="model weight path"
    )
    parser.add_argument(
        "--model_config",default="exp/79_0.1_10/model.yml", type=str, help="model config path"
    )
    parser.add_argument(
        "--data_config", default="exp/79_0.1_10/data.yml", type=str, help="data config used for training."
    )
    parser.add_argument(
        "--run_name", default="decompose", type=str, help="run name for wandb"
    )
    parser.add_argument(
        "--save_name", default="temp", type=str, help="save name"
    )

    args = parser.parse_args()
    data_config = read_yaml(cfg=args.data_config)

    model_instance = Model(args.model_config, verbose=False)
    model_instance.model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl.set_backend('pytorch')
  
    study = optuna.create_study(directions=['maximize','minimize'],pruner=optuna.pruners.MedianPruner(
    n_startup_trials=5, n_warmup_steps=5, interval_steps=5))
    study.optimize(func=Objective(model_instance, data_config, args.run_name), n_trials=100)
    joblib.dump(study, '/opt/ml/code/decomposition_optuna.pkl')