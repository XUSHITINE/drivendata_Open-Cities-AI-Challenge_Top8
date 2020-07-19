'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-01 21:39:47
Description : 
'''
import sys
sys.path.append('..')
from data.dataloader import make_dataloader
from configs import merage_from_arg
from model import bulid_model
from argparse import ArgumentParser
import torch.nn as nn
import torch
from model import bulid_model
torch.backends.cudnn.benchmark = True
import os
from tqdm import tqdm
import cv2 as cv
from configs import merage_from_arg,load_arg
import datetime
import zipfile

def zipDir(dirpath,outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName,"w",zipfile.ZIP_DEFLATED)
    for path,dirnames,filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath,'')

        for filename in filenames:
            zip.write(os.path.join(path,filename),os.path.join(fpath,filename))
    zip.close()

def inference_forward(cfg,model,dataloader,tta_flag=False):
    print("------- start -------")
    device = cfg['device_ids'][0]
    model = model.cuda(device)
    model.eval()
    

    with torch.no_grad():
        for image,filename in tqdm(dataloader):
            image = image.cuda(device)
            predict = model(image)

            if tta_flag:
                predict_flip = model(torch.flip(image,[-1]))
                predict_flip = torch.flip(predict_flip,[-1])
                predict += predict_flip
                predict_flip = model(torch.flip(image,[-2]))
                predict_flip = torch.flip(predict_flip,[-2])
                predict_flip += predict_flip
                predict_flip = model(torch.flip(image,[-1,-2]))
                predict_flip = torch.flip(predict_flip,[-1,-2])
                predict += predict_flip
            predict = torch.argmax(predict.cpu(),1).byte().numpy()
            
            batch = predict.shape[0]
            save_dir = cfg['save_dir']
            for i in range(batch):
                # import pdb; pdb.set_trace()
                cv.imwrite(os.path.join(save_dir,filename[i].replace('.npy','.tif')),predict[i])
                
                

    
if __name__ == "__main__":
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = vars(load_arg())
    if arg['MODEL.LOAD_PATH'] != None: #优先级：arg传入命令 >model中存的cfg > config_file
        cfg = torch.load(arg['MODEL.LOAD_PATH'])['cfg']
    # 待修改
    config_file = arg["CONFIG_FILE"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')
    exec(r"from {} import config as cfg".format(config_file))

    cfg = merage_from_arg(cfg,arg)


    model = bulid_model(cfg['model'],cfg['pretrain'])
    # 默认开启多GPU
    if cfg['multi_gpu']:
        device_ids = cfg['device_ids']
        model = nn.DataParallel(model,device_ids=device_ids)

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_')
    model_tag = (os.path.split(arg['MODEL.LOAD_PATH'])[1]).split('.')[0]
    save_dir = os.path.join(r'../exp/',time_str+model_tag)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("Save_dir :",save_dir)
    cfg['save_dir'] = save_dir
    dataloader = make_dataloader(cfg['test_pipeline'])
    inference_forward(cfg,model,dataloader,tta_flag=True)
    zipDir(save_dir,save_dir+".zip") # 压缩至根目录