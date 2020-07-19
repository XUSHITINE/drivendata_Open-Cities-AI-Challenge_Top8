'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-14 23:35:18
Description : 
'''
from  torch.utils.data import Dataset
from glob import glob
import cv2 as cv
import pdb
import os
import numpy as np
import sys

class load_dataAll(Dataset):
    def __init__(self,train_1_image_dir,train_1_mask_dir,train_2_image_dir=None,train_2_mask_dir=None,
                pseudo_image_dir=None,pseudo_mask_dir=None,extra_image_dir_list=None,extra_mask_dir_list=None,transforms=None):
        
        self.transforms = transforms
        
        self.image_list = list()
        self.mask_list = list()
        # train_1 默认加载，其它数据集看情况加载
        image_list,mask_list = self.glob_from_dir(train_1_image_dir,train_1_mask_dir,end_with="*")

        ################
        print("临时插入  train1 list 1/3 下采样")
        image_list = image_list[:-1:3]
        mask_list = mask_list[:-1:3]
        ################
        print("length of train_1 is ",len(image_list))
        self.image_list += image_list
        self.mask_list += mask_list

        # train_2
        image_list,mask_list = self.glob_from_dir(train_2_image_dir,train_2_mask_dir,end_with="*")
        print("length of train_2 is ",len(image_list))
        self.image_list += image_list
        self.mask_list += mask_list

        # pseudo_label
        image_list,mask_list = self.glob_from_dir(pseudo_image_dir,pseudo_mask_dir)
        print("length of pseudo_label is ",len(image_list))
        self.image_list += image_list
        self.mask_list += mask_list

        # extra_dataset
        if extra_image_dir_list and extra_mask_dir_list:
            for i,(extra_image_dir,extra_mask_dir) in enumerate(zip(extra_image_dir_list,extra_mask_dir_list)):
                image_list,mask_list = self.glob_from_dir(extra_image_dir,extra_mask_dir)
                print("length of extra_dataset_" +str(i) +" is ",len(image_list))
                self.image_list += image_list
                self.mask_list += mask_list
            
    def __len__(self):
        return len(self.image_list)

    def glob_from_dir(self,image_dir,mask_dir,end_with="*"):
        image_list,mask_list = list(),list()
        if image_dir and mask_dir:
            image_list = glob(os.path.join(image_dir,end_with))
            mask_list   = glob(os.path.join(mask_dir,end_with))
            assert len(image_list)==len(mask_list)
            image_list.sort()
            mask_list.sort()
        return image_list,mask_list
    
    def __getitem__(self,idx):
        image_name = self.image_list[idx]
        if ".npy" == image_name[-4:]:
            image = np.load(image_name)
        else:
            image = cv.imread(image_name)
        
        mask_name = self.mask_list[idx]
        if ".npy" == mask_name[-4:]:
            mask = np.load(mask_name)%254
        else:
            mask = cv.imread(mask_name,cv.IMREAD_GRAYSCALE)%254

        sample = {'image':image,'mask':mask}
        if self.transforms:
            sample = self.transforms(sample)
        return sample['image'],sample['mask']




            


class inference_dataset(Dataset):
    def __init__(self,test_image_dir,transforms=None):
        self.transforms = transforms
        self.image_dir = test_image_dir
        self.image_name_list = os.listdir(test_image_dir)

    def __len__(self):
        return len(os.listdir(self.image_dir))
    
    def __getitem__(self,idx):
        image_name = os.path.join(self.image_dir,self.image_name_list[idx])
        image = np.load(image_name)

        mask = np.zeros((image.shape))
        sample = {'image':image,'mask':mask}

        if self.transforms:
            sample = self.transforms(sample)
        return sample['image'],self.image_name_list[idx]




        
if __name__ == "__main__":
    import pdb; pdb.set_trace()