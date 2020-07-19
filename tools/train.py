'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-01 01:04:37
Description : 
'''
import sys
sys.path.append('..')
from data.dataloader import make_dataloader
from configs import merage_from_arg,load_arg
from model import bulid_model,DataParallel_withLoss
from solver import make_optimizer,make_lr_scheduler,make_criterion
from solver import DP_SWA
from argparse import ArgumentParser
import torchcontrib
from engine import do_train
import torch.nn as nn
import torch




if __name__ == "__main__":
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = vars(load_arg())
    # 待修改
    config_file = arg["CONFIG_FILE"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')
    exec(r"from {} import config as cfg".format(config_file))
    # if arg['MODEL.LOAD_PATH'] != None: #优先级：arg传入命令 >model中存的cfg > config_file
    #     cfg = torch.load(arg['MODEL.LOAD_PATH'])['cfg']
    cfg = merage_from_arg(cfg,arg)


    train_dataloader = make_dataloader(cfg['train_pipeline'])
    model = bulid_model(cfg['model'],cfg['pretrain'])
    criterion = make_criterion(cfg['criterion'])
    optimizer = make_optimizer(cfg['optimizer'],model)
    lr_scheduler = make_lr_scheduler(cfg['lr_scheduler'],optimizer)

    if cfg['enable_swa']:  # enable swa，swa需在lr_scheduler之后启动
        optimizer = torchcontrib.optim.SWA(optimizer)
        # optimizer = DP_SWA(optimizer)

    if cfg['multi_gpu']:
        # model = nn.DataParallel(model,device_ids=cfg['device_ids'])
        device_ids=cfg['device_ids']
        model = DataParallel_withLoss(model,criterion,device_ids=device_ids)

    if cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True
    
    do_train(cfg,model=model,train_loader=train_dataloader,val_loader=None,optimizer=optimizer,
                scheduler=lr_scheduler,loss_fn=criterion,metrics=None)
