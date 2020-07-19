'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-01 10:46:04
Description : 
'''
from argparse import ArgumentParser

def load_arg():
    parser = ArgumentParser(description="Pytorch Training")
    parser.add_argument("-config_file","--CONFIG_FILE",type=str,required=False,help="Path to config file")
    parser.add_argument("-multi_gpu","--ENABLE_MULTI_GPU",action="store_true")
    parser.add_argument("-tag","--TAG",type=str)
    parser.add_argument("-device","--DEVICE",type=int,nargs='+',
                        help="list of device_id, e.g. [0,1,2]")

    # DATA
    parser.add_argument("-train_num_workers", "--DATA.DATALOADER.TRAIN_NUM_WORKERS",type=int,
                        help='Num of data loading threads. ')
    parser.add_argument("-val_num_workers", "--DATA.DATALOADER.VAL_NUM_WORKERS",type=int,
                        help='Num of data loading threads. ')
    parser.add_argument("-train_batch_size","--DATA.DATALOADER.TRAIN_BATCH_SIZE",type=int,
                        help="input batch size for training")
    parser.add_argument("-val_batch_size","--DATA.DATALOADER.VAL_BATCH_SIZE",type=int,
                        help="input batch size for validation ")
    parser.add_argument("-val_image_dir","--DATA.DATALOADER.VAL_IMAGE_DIR",type=str)


    # MODEL
    parser.add_argument('-model',"--MODEL.NET_NAME",type=str,
                        help="Net to build")
    parser.add_argument('-path',"--MODEL.LOAD_PATH",type=str,
                        help="path/file of a pretrain model and config ({'model':state_dict(),'cfg':cfg})")

    # SOLVER
    # parser.add_argument("-max_epochs","--SOLVER.MAX_EPOCHS",type=int,
    #                     help="num of epochs to train (default:50)")
    # parser.add_argument('-optimizer',"--SOLVER.OPTIMIZER_NAME",type=str,
    #                     help="optimizer (default:SGD)")
    # parser.add_argument("-criterion","--SOLVER.CRITERION",type=str,
    #                     help="Loss Function ")
    # parser.add_argument("-lr","--SOLVER.LEARNING_RATE",type=float,
    #                     help="Learning rate ")
    # parser.add_argument("-lr_scheduler","--SOLVER.LR_SCHEDULER",type=str)


    # UTILS
    parser.add_argument("-find_lr",action="store_true")
    
    arg = parser.parse_args()
    return arg


    
    
def merage_from_arg(config,arg): # --> dict{},dict{}
    if arg['ENABLE_MULTI_GPU']:
        config['multi_gpu'] = arg['ENABLE_MULTI_GPU']
    if arg['TAG']:
        config['tag'] = arg['TAG']
    else:
        config['tag'] = (((arg['CONFIG_FILE']).split('/')[-1]).split('.'))[0]
    print("TAG : ",config['tag'])
    if arg['DEVICE']:
        config['device_ids'] = arg['DEVICE']
        
    if arg['DATA.DATALOADER.TRAIN_NUM_WORKERS']:
        config['train_pipeline']['num_workers'] = arg['DATA.DATALOADER.TRAIN_NUM_WORKERS']
    if arg['DATA.DATALOADER.TRAIN_BATCH_SIZE']:
        config['train_pipeline']['batch_size'] = arg['DATA.DATALOADER.TRAIN_BATCH_SIZE']
    if arg['DATA.DATALOADER.VAL_NUM_WORKERS']:
        config['test_pipeline']['num_workers'] = arg['DATA.DATALOADER.VAL_NUM_WORKERS']
    if arg['DATA.DATALOADER.VAL_BATCH_SIZE']:
        config['test_pipeline']['batch_size'] = arg['DATA.DATALOADER.VAL_BATCH_SIZE']
    if  arg['DATA.DATALOADER.VAL_IMAGE_DIR']:
        config['test_pipeline']['dataset']['test_image_dir'] = arg['DATA.DATALOADER.VAL_IMAGE_DIR']

    if arg['MODEL.NET_NAME']:
        config['model']['type'] = arg['MODEL.NET_NAME']
    if arg['MODEL.LOAD_PATH'] != None:
        config['pretrain'] = arg['MODEL.LOAD_PATH']
    if arg['find_lr']:
        config['find_lr'] = arg['find_lr']
    return config

if __name__ == "__main__":
    pass