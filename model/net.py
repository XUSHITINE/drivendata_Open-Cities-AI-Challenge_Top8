import torch
import torch.nn as nn
from model.HRNet.models.seg_hrnet import get_seg_model
from model.HRNet.hrnet_config import config

class hrnet_w18_up4(nn.Module):
    def __init__(self,num_classes=2,freeze_num_layers=0):
        super(hrnet_w18_up4,self).__init__()
        self.num_classes = num_classes
        self.freeze_num_layers = freeze_num_layers

        config.merge_from_file(r"../model/HRNet/hrnet_config/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")
        self.hrnet_w18 = get_seg_model(config)
        state_dict = self.hrnet_w18.state_dict()
        
        # pretrain model
        pretrain_state_dict = torch.load(r"../model/HRNet/models/pretrain_model/hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.pth",map_location='cpu')
        keys = list(pretrain_state_dict.keys())

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for item in keys:
            if item[6:] in state_dict.keys() and 'last_layer' not in item:
                # print(item)
                new_state_dict[item[6:]] = pretrain_state_dict.pop(item)
        

        state_dict.update(new_state_dict)
        self.hrnet_w18.load_state_dict(state_dict)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)


        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=120,
                out_channels=60,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=60,
                out_channels=self.num_classes,
                kernel_size=3,
                stride=1,
                padding=1))
    
        self.freeze_layers(self.freeze_num_layers)

    def forward(self,x):
        x = self.hrnet_w18(x)
        x = self.up2(x)
        x = self.last_layer(x)
        return self.up2(x)
    
    def freeze_layers(self,num_layers=9): #默认冻结前9层
        if num_layers <= 0:
            pass
        else:
            for i,(name,child) in enumerate(self.hrnet_w18.named_children()):
                if i < num_layers:
                    print("freeze layer : ",name)
                    for param in child.parameters():
                        param.requires_grad = False
                        i += 1
            
            # conv1
            # bn1
            # conv2
            # bn2
            # relu
            # layer1
            # transition1
            # stage2
            # transition2
            # stage3
            # transition3
            # stage4
            # last_layer

class hrnet_w18_up8(nn.Module):
    def __init__(self,num_classes=2,freeze_num_layers=0):
        super(hrnet_w18_up8,self).__init__()
        self.num_classes = num_classes
        self.freeze_num_layers = freeze_num_layers

        config.merge_from_file(r"../model/HRNet/hrnet_config/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")
        self.hrnet_w18 = get_seg_model(config)
        state_dict = self.hrnet_w18.state_dict()
        
        # pretrain model
        pretrain_state_dict = torch.load(r"../model/HRNet/models/pretrain_model/hrnet_w18_small_v1_cityscapes_cls19_1024x2048_trainset.pth",map_location='cpu')
        keys = list(pretrain_state_dict.keys())

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for item in keys:
            if item[6:] in state_dict.keys() and 'last_layer' not in item:
                # print(item)
                new_state_dict[item[6:]] = pretrain_state_dict.pop(item)
        

        state_dict.update(new_state_dict)
        self.hrnet_w18.load_state_dict(state_dict)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)


        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=120,
                out_channels=60,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=60,
                out_channels=self.num_classes,
                kernel_size=3,
                stride=1,
                padding=1))
    
        self.freeze_layers(self.freeze_num_layers)

    def forward(self,x):
        x = self.hrnet_w18(x)
        x = self.up4(x)
        x = self.last_layer(x)
        return self.up2(x)
    
    def freeze_layers(self,num_layers=9): #默认冻结前9层
        if num_layers <= 0:
            pass
        else:
            for i,(name,child) in enumerate(self.hrnet_w18.named_children()):
                if i < num_layers:
                    print("freeze layer : ",name)
                    for param in child.parameters():
                        param.requires_grad = False
                        i += 1






class hrnet_w48_up4(nn.Module):
    def __init__(self,num_classes=2,freeze_num_layers=0):
        super(hrnet_w48_up4,self).__init__()
        self.num_classes = num_classes
        self.freeze_num_layers = freeze_num_layers

        config.merge_from_file(r"../model/HRNet/hrnet_config/seg_hrnet_w48_train_ohem_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")
        self.hrnet_w48 = get_seg_model(config)
        state_dict = self.hrnet_w48.state_dict()
        
        # pretrain model
        pretrain_state_dict = torch.load(r"../model/HRNet/models/pretrain_model/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth",map_location='cpu')
        keys = list(pretrain_state_dict.keys())

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for item in keys:
            if item[6:] in state_dict.keys() and 'last_layer' not in item:
                new_state_dict[item[6:]] = pretrain_state_dict.pop(item)
        state_dict.update(new_state_dict)
        self.hrnet_w48.load_state_dict(state_dict)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=360,
                out_channels=180,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=180,
                out_channels=self.num_classes,
                kernel_size=3,
                stride=1,
                padding=1))
        self.freeze_layers(num_layers=self.freeze_num_layers)
    def forward(self,x):
        x = self.hrnet_w48(x)
        x = self.up2(x)
        x = self.last_layer(x)
        return self.up2(x)

    def freeze_layers(self,num_layers=9): #默认冻结前9层
        if num_layers <= 0:
            pass
        else:
            for i,(name,child) in enumerate(self.hrnet_w48.named_children()):
                if i < num_layers:
                    print("freeze layer : ",name)
                    for param in child.parameters():
                        param.requires_grad = False
                        i += 1

class hrnet_w48_up8(nn.Module):
    def __init__(self,num_classes=2,freeze_num_layers=0):
        super(hrnet_w48_up8,self).__init__()
        self.num_classes = num_classes
        self.freeze_num_layers = freeze_num_layers

        config.merge_from_file(r"../model/HRNet/hrnet_config/seg_hrnet_w48_train_ohem_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")
        self.hrnet_w48 = get_seg_model(config)
        state_dict = self.hrnet_w48.state_dict()
        
        # pretrain model
        pretrain_state_dict = torch.load(r"../model/HRNet/models/pretrain_model/hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth",map_location='cpu')
        keys = list(pretrain_state_dict.keys())

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for item in keys:
            if item[6:] in state_dict.keys() and 'last_layer' not in item:
                new_state_dict[item[6:]] = pretrain_state_dict.pop(item)
        state_dict.update(new_state_dict)
        self.hrnet_w48.load_state_dict(state_dict)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up4 = nn.Upsample(scale_factor=4)

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=360,
                out_channels=180,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=180,
                out_channels=self.num_classes,
                kernel_size=3,
                stride=1,
                padding=1))
        self.freeze_layers(num_layers=self.freeze_num_layers)
    def forward(self,x):
        x = self.hrnet_w48(x)
        x = self.up4(x)
        x = self.last_layer(x)
        return self.up2(x)

    def freeze_layers(self,num_layers=9): #默认冻结前9层
        if num_layers <= 0:
            pass
        else:
            for i,(name,child) in enumerate(self.hrnet_w48.named_children()):
                if i < num_layers:
                    print("freeze layer : ",name)
                    for param in child.parameters():
                        param.requires_grad = False
                        i += 1
   
        


if __name__ == "__main__":
    
    # model = hrnet_w18()
    # inputs = torch.randn(1,3,256,256)
    # outputs = model(inputs)
    # print(outputs.shape)

    # model = hrnet_w48_up4()
    # inputs = torch.randn(1,3,256,256)
    # outputs = model(inputs)
    # print(outputs.shape)
    pass

