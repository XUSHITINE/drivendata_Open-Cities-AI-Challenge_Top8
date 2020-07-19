'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-03-01 01:04:11
'''
from . import opencv_transforms as transforms


def build_transforms(cfg_transforms):
    cfg_transforms = cfg_transforms.copy()
    transforms_list = list()
    for item in cfg_transforms:
        transforms_type = item.pop('type')
        kwags = item
        transforms_list.append(getattr(transforms,transforms_type)(**kwags))
    print(transforms_list)
    return transforms.Compose(transforms_list)
