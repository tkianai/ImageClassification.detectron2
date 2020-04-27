'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 17:01:36
@FilePath       : /ImageCls.detectron2/imgcls/data/classification_utils.py
@Description    : 
'''


import detectron2.data.transforms as T
import logging


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    input_size = cfg.MODEL.CLSNET.INPUT_SIZE

    logger = logging.getLogger("detectron2.data.classification_utils")
    tfm_gens = []
    tfm_gens.append(T.Resize((input_size, input_size)))
    if is_train:
        tfm_gens.append(T.RandomContrast(0.5, 1.5))
        tfm_gens.append(T.RandomBrightness(0.5, 1.5))
        tfm_gens.append(T.RandomSaturation(0.5, 1.5))
        tfm_gens.append(T.RandomFlip())
        logger.info(
            "TransformGens used in training[Updated]: " + str(tfm_gens))
    return tfm_gens
