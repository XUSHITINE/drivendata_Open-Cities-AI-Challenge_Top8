'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-03-14 23:22:36
Description : 
'''

config = dict(
    # Basic Config
    log_period = 0.01,
    tag = "",
    find_lr = False,
    max_epochs = 24 + 1,
    # save_per_step = 100, # 最好是设置为 swa 中 step_size_up 的偶数倍数
    save_dir = r"../output/model/",
    enable_backends_cudnn_benchmark = True,
    # Dataset
    train_pipeline = dict(
        transforms = [
                    dict(type="RandomHorizontalFlip",p=0.2),
                    dict(type="RandomVerticalFlip",p=0.2),
                    dict(type="ColorJitter",brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),
                    dict(type="Shift_Padding",p=0.1,hor_shift_ratio=0.05,ver_shift_ratio=0.05,pad=0),
                    dict(type="RandomErasing",p=0.2,sl=0.02,sh=0.4,rl=0.2),
                    dict(type="GaussianBlur",p=0.2,radiu=2),
                    dict(type="ToTensor",),
                    dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=False),
                    ],
        dataset = dict(type="load_dataAll",
                    # train_1
                    train_1_image_dir = r"/home/LinHonghui/Datasets/SegBulid/train1/image_bin",train_1_mask_dir = r"/home/LinHonghui/Datasets/SegBulid/train1/label_bin", 
                    # train_2
                    train_2_image_dir = r"",train_2_mask_dir = r"", 
                    # pseudo_label
                    pseudo_image_dir = r"/home/LinHonghui/Datasets/SegBulid/test_dataset_npy/image_bin/", pseudo_mask_dir = r"/home/LinHonghui/Project/DrivenData_2020_SegBulid/exp/new",
                    # extra_datasets --> List[]
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
                    test_image_dir = r"/home/LinHonghui/Datasets/SegBulid/test_dataset_npy/image_bin/"
                    ),
        batch_size = 16,
        shuffle = False,
        num_workers = 8,
        drop_last = False,  
    ),


    # Model
    model = dict(
        type = "hrnet_w48_up4",
        num_classes = 2,
        freeze_num_layers = 9,
    ),
    pretrain = r"",
    multi_gpu = True,
    device_ids = [0,1,2], # 默认第一位作为主卡

    # Solver
    enable_swa = True,
    criterion = dict(type="cross_entropy2d"),
    lr_scheduler = dict(type="CyclicLR",base_lr=1e-6,max_lr=1e-2,step_size_up=10000,mode='triangular2',cycle_momentum=True), # cycle_momentum=False if optimizer==Adam
    optimizer = dict(type="SGD",lr=1e-4,momentum=0.9,weight_decay=1e-5),
    

)


if __name__ == "__main__":

    pass