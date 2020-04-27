###
 # @Copyright (c) tkianai All Rights Reserved.
 # @Author         : tkianai
 # @Github         : https://github.com/tkianai
 # @Date           : 2020-04-27 16:54:32
 # @FilePath       : /ImageCls.detectron2/scripts/train_net_builtin.sh
 # @Description    : 
 ###


CUDA_VISIBLE_DEVICES=4,5,6,7 python train_net_builtin.py --num-gpus 4 --config-file config/Base_image_cls.yaml