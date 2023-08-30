from typing import Any, Optional, Union

import numpy as np
import torch
from mmengine import print_log, MODELS
from mmengine.analysis import FlopAnalyzer
from mmengine.analysis.print_helper import _format_size
from mmengine.hooks import Hook
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import HOOKS
import time
import logging
import os.path as osp

from mmseg.engine.hooks import LogisticWeightPruningHook
from mmseg.models import BaseSegmentor
from mmseg.models.utils import set_identity_layer_mode
from mmseg.models.utils.identity_conv import EmptyModule
from mmseg.models.utils.mask_wrapper import rgetattr, rsetattr
from mmseg.structures import SegDataSample
import torch_pruning as tp


@HOOKS.register_module()
class FLOPSMeasureHook(Hook):
    """
    """
    priority = "VERY_LOW"
    def __init__(self, model_cfg, interval=1000, input_shape=(512,512)):
        self.interval = interval
        self.model_cfg = model_cfg

        if len(input_shape) == 1:
            self.input_shape = (3, input_shape[0], input_shape[0])
        elif len(input_shape) == 2:
            self.input_shape = (3,) + input_shape
        else:
            raise ValueError('invalid input shape')

        result = {}

        result['ori_shape'] = self.input_shape[-2:]
        result['pad_shape'] = self.input_shape[-2:]
        self.result = result

    def get_data_batch(self):
        return {
            'inputs': [torch.rand(self.input_shape)],
            'data_samples': [SegDataSample(metainfo=self.result)]
        }

    def _delete_remove_layers(self, history, model):
        for i in range(len(history) - 1, -1, -1):
            name = history[i]
            if "delete " in name:
                name = name.replace("delete ", "")
                rsetattr(model, name, EmptyModule())
                del history[i]
    def reload_model(self, history, model):
        self._delete_remove_layers(history, model)
        set_identity_layer_mode(model, True)
        def _forward_func(self, data):
            return self(data)

        def _output_transform(data):
            l = []
            for i in data:
                l.append(i)
            return l

        data_batch = model.module.data_preprocessor(self.get_data_batch())
        DG = tp.DependencyGraph().build_dependency(model,
                                                   example_inputs=data_batch['inputs'],
                                                   forward_fn=_forward_func,
                                                   output_transform=_output_transform)
        DG.load_pruning_history(history)
        for module_name, module in model.named_modules():
            if getattr(module, "mask_class_wrapper", False):
                module.reinit_masks()
        set_identity_layer_mode(model, False)
    def measure_flops(self, runner):
        #model = revert_sync_batchnorm(model)
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.module = model

            def forward(self, *args, **kw):
                return self.module(*args, **kw)
        model = ModelWrapper(MODELS.build(self.model_cfg))
        model.eval()
        log_hook = [h for h in runner._hooks if isinstance(h, LogisticWeightPruningHook)]
        if len(log_hook) > 0:
            log_hook = log_hook[0]
            history = log_hook.history
            self.reload_model(history, model)
        #model.eval()
        with torch.no_grad():
            data = model.module.data_preprocessor(self.get_data_batch())
            flop_analyzer = FlopAnalyzer(model, data['inputs'])
            flops = flop_analyzer.total()
            del flop_analyzer
        del model
        return _format_size(flops)


    def print_flops(self, flops):
        print_log("\n_____________________________\n"
                  f"FLOPS {flops}"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.INFO)

    def before_train_epoch(self, runner) -> None:
        self.print_flops(self.measure_flops(runner))

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs: Optional[dict] = None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            #try:
            self.print_flops(self.measure_flops(runner))
            #except:
            #    pass


