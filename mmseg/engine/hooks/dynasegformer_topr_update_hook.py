from typing import  Optional, Union


from mmengine.hooks import Hook
from mmengine.registry import HOOKS



DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class DynaSegFormerTopRUpdateHook(Hook):
    """
    """

    def __init__(self, min_topr, sparsity_annealing_steps):
        self.min_topr = min_topr
        self.sparsity_annealing_steps = sparsity_annealing_steps

    def set_topr_dgl(self, value, model):
        for module_name, module in model.named_modules():
            if getattr(module, "set_topr_dgl", False):
                module.set_topr_dgl(value)

    def before_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:

        iter = runner.iter
        model = runner.model
        sparsity = self.min_topr * min(1.0, iter/self.sparsity_annealing_steps)
        topr_t = 1 - sparsity
        self.set_topr_dgl(topr_t, model)


