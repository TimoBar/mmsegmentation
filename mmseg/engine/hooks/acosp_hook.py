from typing import Any, Optional, Union

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

import acosp

DATA_BATCH = Optional[Union[dict, tuple, list]]
from acosp.pruner import SoftTopKPruner
import acosp.inject

@HOOKS.register_module()
class AcospHook(Hook):
    """
    """

    def __init__(self, max_iters, interval=1000):
        self.interval = interval
        self.max_epochs = max_iters // interval
        self.pruner = SoftTopKPruner(
            starting_epoch=0,
            ending_epoch=self.max_epochs,  # Pruning duration
            final_sparsity=0.5,  # Final sparsity
        )

    def before_run(self, runner) -> None:
        self.pruner.configure_model(runner.model.cuda())

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            self.pruner.update_mask_layers(runner.model, runner.iter // self.interval)
        if self.is_last_train_iter(runner):
            acosp.inject.soft_to_hard_k(runner.model)

