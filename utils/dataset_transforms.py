import torchvision
import torch
import torchvision.transforms as transforms

def read_transforms(transformCfg):
    transform_list = []
    for param in transformCfg:
        if param.type == 'Resize':
            transform_list.append(transforms.Resize(param.size,antialias=True))
        elif param.type == 'CenterCrop':
            transform_list.append(transforms.CenterCrop(param.size))
        elif param.type == 'RandomHorizontalFlip':
            transform_list.append(transforms.RandomHorizontalFlip(param.p))
        elif param.type == 'ToTensor':
            transform_list.append(transforms.ToTensor())
        elif param.type == 'Normalize':
            transform_list.append(transforms.Normalize(param.mean, param.std))
        elif param.type == 'CenterCrop':
            transform_list.append(transforms.CenterCrop(param.size))

    transform = transforms.Compose(transform_list)
    return transform