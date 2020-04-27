'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 14:26:03
@FilePath       : /ImageCls.detectron2/imgcls/config/defaults.py
@Description    : 
'''


from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# MobileNets
# ---------------------------------------------------------------------------- #
_C.MODEL.MNET = CN()

# Output features
_C.MODEL.MNET.OUT_FEATURES = ['linear']
# Width mult
_C.MODEL.MNET.WIDTH_MULT = 1.0


# ---------------------------------------------------------------------------- #
# ClsNets
# ---------------------------------------------------------------------------- #
_C.MODEL.CLSNET = CN()
_C.MODEL.CLSNET.ENABLE = False
# classes number
_C.MODEL.CLSNET.NUM_CLASSES = 1000
# In features
_C.MODEL.CLSNET.IN_FEATURES = ['linear']
# Input Size
_C.MODEL.CLSNET.INPUT_SIZE = 224
