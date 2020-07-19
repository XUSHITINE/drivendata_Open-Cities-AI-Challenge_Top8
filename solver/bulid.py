'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors : now more
Description : 
LastEditTime: 2019-12-09 16:22:49
'''


import torch
import solver.criterion as criterion
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torchcontrib

def make_criterion(cfg_criterion):
    cfg_criterion = cfg_criterion.copy()
    criterion_type = cfg_criterion.pop('type')
    if hasattr(torch.nn,criterion_type):
        loss_fun = getattr(torch.nn,)(**cfg_criterion)
    elif hasattr(criterion,criterion_type):
        loss_fun = getattr(criterion,criterion_type)(**cfg_criterion)
    else:
        print("Loss function not found. Got {}".format(criterion_type))
        import pdb; pdb.set_trace()
    return loss_fun

def make_lr_scheduler(cfg_lr_scheduler,optimizer):
    cfg_lr_scheduler = cfg_lr_scheduler.copy()
    lr_scheduler_type = cfg_lr_scheduler.pop('type')
    if hasattr(lr_scheduler,lr_scheduler_type):
        lr = getattr(lr_scheduler,lr_scheduler_type)(optimizer,**cfg_lr_scheduler)
        return lr
    else:
        print("lr_scheduler not found. Got {}".format(lr_scheduler_type))
        import pdb; pdb.set_trace()

def make_optimizer(cfg_optimizer,model):
    cfg_optimizer = cfg_optimizer.copy()
    optimizer_type = cfg_optimizer.pop('type')
    if hasattr(optim,optimizer_type):
        params = model.parameters()
        optimizer = getattr(optim,optimizer_type)(params,**cfg_optimizer)
        return optimizer
    else:
        print("optimizer not found. Got {}".format(optimizer_type))
        import pdb; pdb.set_trace()