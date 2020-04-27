'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 14:18:06
@FilePath       : /ImageCls.detectron2/imgcls/config/__init__.py
@Description    : 
'''

from detectron2.config import CfgNode


__all__ = ['get_cfg']


def get_cfg() -> CfgNode:
    """Get a copy of the default config

    Returns:
        CfgNode -- a detectron2 CfgNode instance
    """

    from .defaults import _C
    return _C.clone()
    

