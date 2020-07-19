'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-02-28 17:38:36
Description : 
'''
import data.dataset.bulid as datasets

def build_dataset(cfg_dataset,transforms):
    dataset_type = cfg_dataset.pop('type')
    dataset_kwags = cfg_dataset
    dataset = getattr(datasets,dataset_type)(**dataset_kwags,transforms=transforms)
    return dataset

# def build_dataset(cfg,transforms=None,is_train=True):
#     '''
#     Description: build_dataset
#     Args (type): 
#         cfg (yaml): config file.
#         transforms (callable,optional): Optional transforms to be applied on a sample.
#         is_train (bool): True or False.
#     return: 
#         dataset(torch.utils.data.Dataset)
#     '''
#     DATASET = cfg.DATA.DATASET
#     if is_train==True:
#         train_image_dir = DATASET.train_image_dir
#         train_mask_dir = DATASET.train_mask_dir
#         dataset = png_dataset(cfg,train_image_dir,train_mask_dir,transforms)
#     else:
#         test_image_dir = DATASET.test_image_dir
#         dataset = inference_dataset(cfg,test_image_dir,transforms)
#     return dataset