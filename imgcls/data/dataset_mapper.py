'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 17:01:20
@FilePath       : /ImageCls.detectron2/imgcls/data/dataset_mapper.py
@Description    : 
'''

from . import classification_utils as c_utils


from detectron2.data.dataset_mapper import DatasetMapper as _DatasetMapper



class DatasetMapper(_DatasetMapper):

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        self.tfm_gens = c_utils.build_transform_gen(cfg, is_train)