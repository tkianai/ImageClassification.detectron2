###
 # @Copyright (c) tkianai All Rights Reserved.
 # @Author         : tkianai
 # @Github         : https://github.com/tkianai
 # @Date           : 2020-04-27 16:55:24
 # @FilePath       : /ImageCls.detectron2/scripts/train_net.sh
 # @Description    : 
 ###


CUDA_VISIBLE_DEVICES=4,5,6,7 python train_net.py --config-file configs/Base_image_cls.yaml --batch-size 1024 --dist-url 'tcp://127.0.0.1:51151' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 datasets/ImageNet2012
