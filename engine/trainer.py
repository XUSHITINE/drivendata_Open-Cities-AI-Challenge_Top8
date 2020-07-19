import logging
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.engine import Events, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer,TerminateOnNan
from ignite.metrics import  Loss, RunningAverage,Accuracy
import re
import torch
import os
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor

def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_supervised_dp_trainer(model, optimizer,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    """
    Factory function for creating a trainer for supervised models.

    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
            with respect to the host. For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
            tuple of tensors `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

    Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        loss,y_pred = model(x,y)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,loss_fn,metrics):

    device = cfg['device_ids'][0] if torch.cuda.is_available() else 'cpu' #默认主卡设置为
    max_epochs = cfg['max_epochs']

    # create trainer
    if cfg['multi_gpu']: #多卡时，不需要传入loss_fn
        trainer = create_supervised_dp_trainer(model.train(),optimizer,device=device)
    else:
        trainer = create_supervised_trainer(model.train(),optimizer,loss_fn,device=device)
    trainer.add_event_handler(Events.ITERATION_COMPLETED,TerminateOnNan())
    RunningAverage(output_transform=lambda x:x).attach(trainer,'avg_loss')

    # create pbar
    len_train_loader = len(train_loader)
    pbar = tqdm(total=len_train_loader)



    ##########################################################################################
    ###########                    Events.ITERATION_COMPLETED                    #############
    ##########################################################################################

    # 每 log_period 轮迭代结束输出train_loss
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        log_period = cfg['log_period']
        log_per_iter = int(log_period*len_train_loader) if int(log_period*len_train_loader) >=1 else 1   # 计算打印周期
        current_iter = (engine.state.iteration-1)%len_train_loader + 1 + (engine.state.epoch-1)*len_train_loader # 计算当前 iter

        lr = optimizer.state_dict()['param_groups'][0]['lr']

        if current_iter % log_per_iter == 0:
            pbar.write("Epoch[{}] Iteration[{}] lr {:.7f} Loss {:.7f}".format(engine.state.epoch,current_iter,lr,engine.state.metrics['avg_loss']))
            pbar.update(log_per_iter)
    
    # lr_scheduler
    @trainer.on(Events.ITERATION_COMPLETED)
    def adjust_lr_scheduler(engine):
        if isinstance(scheduler,lr_scheduler.CyclicLR):
            scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_swa(engine):
        if isinstance(scheduler,lr_scheduler.CyclicLR):
            if cfg['enable_swa']:
                swa_period = 2*cfg['lr_scheduler']['step_size_up']
                current_iter = (engine.state.iteration-1)%len_train_loader + 1 + (engine.state.epoch-1)*len_train_loader # 计算当前 iter
                if current_iter%swa_period==0:
                    optimizer.update_swa()

    
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def update_bn(engine):
        if isinstance(scheduler,lr_scheduler.CyclicLR):
            save_period = 2*cfg['lr_scheduler']['step_size_up']
            current_iter = (engine.state.iteration-1)%len_train_loader + 1 + (engine.state.epoch-1)*len_train_loader # 计算当前 iter
            if current_iter%save_period==0 and current_iter >= save_period*2:  # 从第 4 个周期开始存
                
                save_dir = cfg['save_dir']
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                if cfg['enable_swa']:
                    optimizer.swap_swa_sgd()
                    optimizer.bn_update(train_loader,model,device=device)
                model_name=os.path.join(save_dir,cfg['model']['type'] + '_' + cfg['tag'] + "_" + str(current_iter) + ".pth")
                if cfg['multi_gpu']:
                    save_pth = {'model':model.module.model.state_dict(),'cfg':cfg}
                    torch.save(save_pth,model_name)
                else:
                    save_pth = {'model':model.state_dict(),'cfg':cfg}
                    torch.save(save_pth,model_name)



    ##########################################################################################
    ##################               Events.EPOCH_COMPLETED                    ###############
    ##########################################################################################
    @trainer.on(Events.EPOCH_COMPLETED)
    def save_temp_epoch(engine):
        save_dir = cfg['save_dir']
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        epoch = engine.state.epoch
        if epoch%1==0:
            model_name=os.path.join(save_dir,cfg['model']['type'] + '_' + cfg['tag'] +"_temp.pth")
            if cfg['multi_gpu']:
                save_pth = {'model':model.module.model.state_dict(),'cfg':cfg}
                torch.save(save_pth,model_name)
            else:
                save_pth = {'model':model.state_dict(),'cfg':cfg}
                torch.save(save_pth,model_name)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def reset_pbar(engine):
        pbar.reset()
    
    
    trainer.run(train_loader,max_epochs=max_epochs)
    pbar.close()