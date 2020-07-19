###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-03-15 14:31:38
 # @Description : 
 ###
 
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/LinHonghui/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/LinHonghui/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

cd tools
conda activate SegBulid

# config_file="../configs/server_6121_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_batch_12_step_size_up_5000.py"  
# config_file="../configs/server_6121_hrnet_w48_up8_freeze_0_sgd_LabelSmoothing_train_1_batch_9_step_size_up_5000.py"
# config_file="../configs/server_6125_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_agri_p79_batch_12_step_size_up_5000.py"
path="/home/LinHonghui/Project/DrivenData_2020_SegBulid/output/model/hrnet_w48_up8_server_6125_hrnet_w48_up8_freeze_9_sgd_cross_entropy2d_train_1_agri_p79_batch_12_step_size_up_5000_30000.pth"
config_file="../configs/server_6121_hrnet_w48_up8_freeze_9_sgd_train_1_p8389_batch_3_step_size_up_5000.py"
python train.py -device 3    -config_file $config_file -path $path

