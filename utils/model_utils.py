import torchvision
import torch
import torchvision.transforms as transforms
from torch import nn



def calculate_convmap_size(input_size, layers):
    '''
    Function for computing pure Convs & Pools Net's feature map size.
    '''
    size = input_size
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            size = (size - layer.kernel_size[0] + 2 * layer.padding[0]) // layer.stride[0] + 1
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
            size = (size - layer.kernel_size) // layer.stride + 1
        else:
            pass
    return size
