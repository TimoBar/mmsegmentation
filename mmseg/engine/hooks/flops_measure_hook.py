from typing import Any, Optional, Union

import numpy as np
import torch
from mmengine import print_log
from mmengine.analysis import FlopAnalyzer
from mmengine.analysis.print_helper import _format_size
from mmengine.hooks import Hook
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import HOOKS
import time
import logging

from mmseg.models import BaseSegmentor
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class FLOPSMeasureHook(Hook):
    """
    """
    priority = 'VERY_LOW'
    def __init__(self, interval=1000, input_shape=(512,512)):
        self.interval = interval

        if len(input_shape) == 1:
            self.input_shape = (3, input_shape[0], input_shape[0])
        elif len(input_shape) == 2:
            self.input_shape = (3,) + input_shape
        else:
            raise ValueError('invalid input shape')

        result = {}

        result['ori_shape'] = self.input_shape[-2:]
        result['pad_shape'] = self.input_shape[-2:]
        self.data_batch = {
            'inputs': [torch.rand(self.input_shape)],
            'data_samples': [SegDataSample(metainfo=result)]
        }


    def measure_flops(self, model):
        model = revert_sync_batchnorm(model)
        model.eval()
        data = model.data_preprocessor(self.data_batch)

        flop_handler = FlopAnalyzer(model, data['inputs'])
        with torch.no_grad():
            flops = flop_handler.total()
        model.train()
        return _format_size(flops)


    def print_flops(self, flops):
        print_log("\n_____________________________\n"
                  f"FLOPS {flops}"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.INFO)

    def before_train(self, runner) -> None:
        self.print_flops(self.measure_flops(runner.model.module))

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs: Optional[dict] = None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            try:
                self.print_flops(self.measure_flops(runner.model.module))
            except:
                # if one layer is completely pruned flop count fails
                pass

