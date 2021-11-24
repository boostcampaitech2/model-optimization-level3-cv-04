import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import make_divisible

class ShuffleNetV2(nn.Module):
    def __init__(self, inp, stride):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        
        if stride == 1 :
            # channel split 
            c1 = inp // 2
            self.branch2 = nn.Sequential(
                            # 1x1 conv
                            nn.Conv2d(c1 , c1 , kernel_size = 1, stride = 1, padding = 0, bias = False),
                            nn.BatchNorm2d(c1),
                            nn.ReLU(True),
                            # 3x3 Depthwise COnv
                            nn.Conv2d(c1,c1,kernel_size = 3, stride = 1, padding = 1, bias = False, groups = c1),
                            nn.BatchNorm2d(c1),
                            # 1x1 conv
                            nn.Conv2d(c1 , c1 , kernel_size = 1, stride = 1, padding = 0, bias = False),
                            nn.BatchNorm2d(c1),
                            nn.ReLU(True)
                            )
        else :
            self.branch1 = nn.Sequential(
                            # 3x3 Depthwise COnv
                            nn.Conv2d(inp, inp, kernel_size = 3, stride = 2, padding = 1, bias = False, groups = inp),
                            nn.BatchNorm2d(inp),
                            # 1x1 conv
                            nn.Conv2d(inp , inp , kernel_size = 1, stride = 1, padding = 0, bias = False),
                            nn.BatchNorm2d(inp),
                            nn.ReLU(True)
                        )
            self.branch2 = nn.Sequential(
                            # 1x1 conv
                            nn.Conv2d(inp , inp , kernel_size = 1, stride = 1, padding = 0, bias = False),
                            nn.BatchNorm2d(inp),
                            nn.ReLU(True),
                            # 3x3 Depthwise COnv
                            nn.Conv2d(inp, inp, kernel_size = 3, stride = 2, padding = 1, bias = False, groups = inp),
                            nn.BatchNorm2d(inp),
                            # 1x1 conv
                            nn.Conv2d(inp , inp , kernel_size = 1, stride = 1, padding = 0, bias = False),
                            nn.BatchNorm2d(inp),
                            nn.ReLU(True)
                    )              


    def forward(self, x: torch.Tensor):
        # channel concat
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.branch2(x2)
            out = torch.cat((x1, x2), dim = 1)
        else :
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            out = torch.cat((x1,x2) , dim = 1)
        
        # channel shuffle
        out = self.channel_shuffle(out , 2)
        return out
        
        
    def channel_shuffle(self, x: Tensor, groups: int) -> Tensor:
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)

        return x        
    
    
    
class ShuffleNetV2Generator(GeneratorAbstract):
    # args 순서 : out_channel , stride
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.in_channel * self.width_multiply * self.args[0])

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        module = []
        s = self.args[0]  # c is equivalent as self.out_channel
        inp, _ = self.in_channel, self.out_channel
        for i in range(repeat):
            module.append(
                self.base_module(
                    inp=inp,
                    stride=s
                )
            )
            inp = inp*s
        return self._get_module(module)