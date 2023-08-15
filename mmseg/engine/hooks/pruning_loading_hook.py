from typing import Any, Optional, Union

import numpy as np
import torch
from mmengine import print_log
from mmengine.analysis import FlopAnalyzer
from mmengine.analysis.print_helper import _format_size
from mmengine.device import get_device
from mmengine.hooks import Hook
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import HOOKS
import time
import os
import os.path as osp
import logging
import pickle

from mmengine.runner.checkpoint import _load_checkpoint, find_latest_checkpoint
import torch_pruning as tp

from mmseg.models import BaseSegmentor
from mmseg.models.utils import set_identity_layer_mode
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class PruningLoadingHook(Hook):
    """
    """
    priority = 'VERY_LOW'
    def __init__(self, interval=1000):
        self.interval = interval

    def find_last_history(self, path):
        save_file = osp.join(path, 'last_history')
        last_saved: Optional[str]
        if os.path.exists(save_file):
            with open(save_file) as f:
                last_saved = f.read().strip()
        else:
            print_log('Did not find last_history to be resumed.')
            last_saved = None
        return last_saved
    def get_data_batch(self, input_shape):
        result = {}

        result['ori_shape'] = input_shape[-2:]
        result['pad_shape'] = input_shape[-2:]
        data_batch = {
            'inputs': [torch.rand(input_shape)],
            'data_samples': [SegDataSample(metainfo=result)]
        }
        return data_batch

    def before_run(self, runner) -> None:

        history_filename = self.find_last_history(runner.work_dir)
        if history_filename is not None:
            with open(history_filename, 'rb') as handle:
                history = pickle.load(handle)

            set_identity_layer_mode(runner.model, True)

            def _forward_func(self, data):
                return self.test_step(data)

            def _output_transform(data):
                l = []
                for i in data:
                    l.append(i.seg_logits.data)
                return l

            # model.eval()
            DG = tp.DependencyGraph().build_dependency(runner.model, example_inputs=self.get_data_batch((1, 3, 512, 512)), forward_fn=_forward_func,
                                                       output_transform=_output_transform)
            DG.load_pruning_history(history)


    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs: Optional[dict] = None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            self.save_history()

