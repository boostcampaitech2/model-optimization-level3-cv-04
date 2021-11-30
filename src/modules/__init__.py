"""PyTorch Module and ModuleGenerator."""

from src.modules.base_generator import GeneratorAbstract, ModuleGenerator
from src.modules.bottleneck import Bottleneck, BottleneckGenerator
from src.modules.conv import Conv, ConvGenerator, FixedConvGenerator
from src.modules.dwconv import DWConv, DWConvGenerator
from src.modules.flatten import FlattenGenerator

from src.modules.invertedresidualv2 import (InvertedResidualv2,
                                            InvertedResidualv2Generator)
from src.modules.invertedresidualv3 import (InvertedResidualv3,
                                            InvertedResidualv3Generator)
from src.modules.shufflenetv2 import (ShuffleNetV2, ShuffleNetV2Generator)
from src.modules.mbconv import (MBConv, MBConvGenerator)
from src.modules.linear import Linear, LinearGenerator
from src.modules.poolings import (AvgPoolGenerator, GlobalAvgPool,
                                  GlobalAvgPoolGenerator, MaxPoolGenerator)
from src.modules.resbottleneck import ResBottleneck, ResBottleneckGenerator

__all__ = [
    "ModuleGenerator",
    "GeneratorAbstract",
    "Bottleneck",
    "Conv",
    "DWConv",
    "Linear",
    "GlobalAvgPool",
    "InvertedResidualv2",
    "InvertedResidualv3",
    "BottleneckGenerator",
    "FixedConvGenerator",
    "ConvGenerator",
    "LinearGenerator",
    "DWConvGenerator",
    "FlattenGenerator",
    "MaxPoolGenerator",
    "AvgPoolGenerator",
    "GlobalAvgPoolGenerator",
    "InvertedResidualv2Generator",
    "InvertedResidualv3Generator",
    "ShuffleNetV2",
    "ShuffleNetV2Generator",
    "MBConv",
    "MBConvGenerator",
    "ResBottleneckGenerator", 
    "ResBottleneck"
]
