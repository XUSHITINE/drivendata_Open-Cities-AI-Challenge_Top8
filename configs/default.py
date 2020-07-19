'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-03 12:55:05
Description : 
'''

config = dict(
    # Basic Config
    log_period = 0.01,  # 默认0.01，loss打印周期，默认0.01*len(epoch)个iter打印一次batch_loss
    tag = "",           # 默认留空，tag用于对存储模型权重打标注备注，默认自动加载配置文件名作为tag
    find_lr = False,    # 开启新实验时候，需要搜索一次学习率，可以不管
    max_epochs = 24 + 1,# 最大epoch次数

    save_dir = r"../output/model/", # 模型权重保存路径
    enable_backends_cudnn_benchmark = True, # 默认开始加速，会损失一点计算精度

    # Dataset
    # train_dataloader：
    ##          --> transforms: 数据增强加载，把需要用但数据增强函数和参数写在此处，将自动调用对应的数据增强函数，
    #                           data/transforms/opencv_transforms.py中存在的类都可直接调用
    ##          --> dataset: 继承自 torch.utils.data.Dataset，加载image和label，封装成 sample,对应代码在 data/dataset/bulid.py
    #                           (e.g. sample = {'image':image_array,"mask":mask_array},dataset中调用transforms执行在线数据增强)
    ##          --> batch_size: 每个batch数目
    ##          --> shuffle : 是否打乱数据，默认训练集打乱，测试集不打乱
    ##          --> num_workers : 多线程加载数据
    ##          --> drop_last : 若 len_epoch 无法整除 batch_size 时，丢弃最后一个batch。（在比较早的torch版本中，开启多卡GPU加速后，
    #                           若batch无法整除多卡数目，代码运行会报错，避免出错风险丢弃最后一个batch）
    train_pipeline = dict( 
        transforms = [
                    dict(type="RandomHorizontalFlip",p=0.5),
                    dict(type="RandomVerticalFlip",p=0.5),
                    dict(type="ColorJitter",brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                    dict(type="Shift_Padding",p=0.1,hor_shift_ratio=0.05,ver_shift_ratio=0.05,pad=0),
                    dict(type="RandomErasing",p=0.2,sl=0.02,sh=0.4,rl=0.2),
                    dict(type="GaussianBlur",p=0.2,radiu=2),
                    dict(type="ToTensor",),
                    dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
                    ],
        dataset = dict(type="load_dataAll",
                    # train_1 数据集1
                    train_1_image_dir = r"/home/chenbangdong/cbd/DrivenDATA/dataset2/image_bin",train_1_mask_dir = r"/home/chenbangdong/cbd/DrivenDATA/dataset2/label_bin", 
                    # train_2 数据集2(含噪声)
                    train_2_image_dir = r"/home/chenbangdong/cbd/DrivenDATA/train_tier2_dataset/image_bin",train_2_mask_dir = r"/home/chenbangdong/cbd/DrivenDATA/train_tier2_dataset/label_bin", 
                    # pseudo_label (伪标签，含噪声)
                    pseudo_image_dir = r"", pseudo_mask_dir = r"",  
                    # extra_datasets --> List[] (外部数据集，若有多个，则按顺序写入list即可)
                    extra_image_dir_list = [r""],  extra_mask_dir_list = [r""],   
        ),
        batch_size = 8,
        shuffle = True,
        num_workers = 8,
        drop_last = True
    ),

    test_pipeline = dict(
        transforms = [
                    dict(type="ToTensor",),
                    dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
        ],
        dataset = dict(type="load_Inference_dataset",
                    test_image_dir = r"/home/chenbangdong/cbd/DrivenDATA/test_dataset_npy/image_bin/"
                    ),
        batch_size = 16,
        shuffle = False,
        num_workers = 8,
        drop_last = False,  
    ),


    # Model
    # model : 
    ##      --> type : 自定义网络，需在model/net.py中定义
    ##      --> num_classes : 预测类别数目
    ##      --> freeze_num_layers : 网络冻结层数，这里默认冻结了HRNet_W48 stage2 之前的全部网络
    ##      --> pretrain : 是否加载训练好的网络，可以在代码运行时用“-path”传入路径
    ##      --> device_ids : 指定显卡id,可在代码运行时，用“-device”命令传入，(e.g. -device 1 2 3 ) 
    ##      --> multi_gpu : 是否开启多卡
    model = dict(
        type = "hrnet_w48_up4",
        num_classes = 2,
        freeze_num_layers = 9,
    ),
    pretrain = r"",
    multi_gpu = True,
    device_ids = [0,1,2], # 默认第一位作为主卡

    # Solver
    ## enable_swa :对应开启SWA训练策略，单卡时可开启，多卡开启会出错，这个bug还没解决
    ## criterion : 指定loss函数，可自定义，自定义loss存于solver/criterion.py
    ## lr_scheduler : 学习率调整策略，默认从 torch.optim.lr_scheduler 中加载
    ## optimizer : 优化器，默认从 torch.optim 中加载
    enable_swa = False,
    criterion = dict(type="cross_entropy2d"),
    lr_scheduler = dict(type="CyclicLR",base_lr=1e-6,max_lr=1e-2,step_size_up=100,mode='triangular2',cycle_momentum=True), # cycle_momentum=False if optimizer==Adam
    optimizer = dict(type="SGD",lr=1e-4,momentum=0.9,weight_decay=1e-5),

    # Output
    # save_dir = r""
    

)


if __name__ == "__main__":

    pass