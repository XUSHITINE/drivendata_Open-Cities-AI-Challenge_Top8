'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-02-28 17:47:44
Description : 
'''
from ..dataset import build_dataset
from torch.utils.data import DataLoader
from ..transform import build_transforms

def make_dataloader(cfg_dataloader):
    cfg_dataloader = cfg_dataloader.copy()
    cfg_dataset = cfg_dataloader.pop('dataset')
    cfg_transform = cfg_dataloader.pop('transforms')
    
    transforms = build_transforms(cfg_transform)
    dataset = build_dataset(cfg_dataset,transforms=transforms)
    dataloader = DataLoader(dataset,**cfg_dataloader)

    return dataloader
