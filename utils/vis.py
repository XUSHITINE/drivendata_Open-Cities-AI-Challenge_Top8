'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-01 04:20:07
Description : 
'''
import cv2 as cv
import os
from glob import glob
from tqdm import tqdm


if __name__ == "__main__":
    test_dir = r""
    predict_dir = r""
    image_name_list = os.listdir(test_dir)
    
    for i in tqdm(image_name_list):
        test_image = cv.imread(os.path.join(test_dir)
    pass