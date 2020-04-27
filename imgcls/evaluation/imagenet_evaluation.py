'''
@Copyright (c) tkianai All Rights Reserved.
@Author         : tkianai
@Github         : https://github.com/tkianai
@Date           : 2020-04-26 19:42:05
@FilePath       : /ImageCls.detectron2/imgcls/evaluation/imagenet_evaluation.py
@Description    : 
'''


import itertools
import json
import logging
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator


class ImageNetEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2.evaluation.imagenet_evaluation")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._gt = json.load(open(json_file))

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            prediction["gt"] = input["label"]
            prediction["pred"] = output["pred_classes"].to(self._cpu_device)
            self._predictions.append(prediction)


    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning(
                "[ImageNetEvaluator] Did not receive valid predictions.")
            return {}

        topk = len(self._predictions[0]['pred'])
        target = []
        pred = []
        for p in self._predictions:
            target.append(p['gt'])
            pred.append(p['pred'])
        pred = torch.stack(pred, dim=0)
        target = torch.as_tensor(target, dtype=pred.dtype)
        num_samples = target.size(0)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        topk_acc = []
        for k in range(1, topk + 1):
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            topk_acc.append(correct_k.mul_(100.0 / num_samples))
        result = OrderedDict(
            accuracy={"top{}".format(i + 1): acc.item() for i, acc in enumerate(topk_acc)},
        )
        return result
