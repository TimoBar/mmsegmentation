import logging
import os
import os.path as osp
import pickle
from typing import Optional, Union

import torch
import torch_pruning as tp
from mmcv.cnn import ConvModule
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.registry import HOOKS

from mmseg.models.utils import set_identity_layer_mode
from mmseg.models.utils.identity_conv import EmptyModule, DummyModule
from mmseg.models.utils.mask_wrapper import LearnableMask, LearnableMaskLinear, \
    LearnableMaskConv2d, rsetattr
from mmseg.structures import SegDataSample

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class PruningHook(Hook):
    """
    Hook that prunes the model at given time points and prints the current pruning progress in a given interval
    """
    priority = 'LOW'

    def __init__(self, do_logging=True, do_explicit_pruning=True, logging_interval=100, pruning_interval=25,
                 debug=False, prune_at_start=False):
        self.logging_interval = logging_interval
        self.pruning_interval = pruning_interval
        self.logging = do_logging
        self.do_explicit_pruning = do_explicit_pruning
        self.remove_total = 0
        self.num_weights_total_first = -1
        self.model_sizes_org = {}
        self.history = []
        self.debug = debug
        self.prune_at_start = prune_at_start

    def find_last_history(self, path):
        """
        get path of pruning last history file
        Args:
            path:

        Returns:

        """
        save_file = osp.join(path, 'last_history')
        if os.path.exists(save_file):
            return save_file
        else:
            print_log('Did not find last_history to be resumed.')
            return None

    def get_data_batch(self, input_shape):
        """
        get dummy data batch for pruning graph computation
        Args:
            input_shape: input shape of image

        Returns:

        """
        result = {}

        result['ori_shape'] = input_shape[-2:]
        result['pad_shape'] = input_shape[-2:]
        result['img_shape'] = input_shape[-2:]
        data_batch = {
            'inputs': [torch.rand(input_shape)],
            'data_samples': [SegDataSample(metainfo=result)]
        }
        return data_batch

    def is_resume(self, runner):
        """
        check whether this training is resumed from a previous checkpoint
        Args:
            runner:

        Returns:

        """
        return runner._resume or runner._load_from

    def before_run(self, runner) -> None:
        history_filename = self.find_last_history(runner.work_dir)

        # load pruning history if training is resumed
        # then based on pruning history, prune the model as it was in the checkpoint
        if history_filename is not None and self.is_resume(runner):
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

            data_batch = runner.model.module.data_preprocessor(self.get_data_batch((3, 512, 512)))
            DG = tp.DependencyGraph().build_dependency(runner.model,
                                                       example_inputs=data_batch,
                                                       forward_fn=_forward_func,
                                                       output_transform=_output_transform)
            DG.load_pruning_history(history)
            self.history.extend(history)
            for module_name, module in runner.model.named_modules():
                if getattr(module, "mask_class_wrapper", False):
                    module.reinit_masks()
            set_identity_layer_mode(runner.model, False)

    def init_model_stats(self, model):
        for n, p in model.named_modules():
            if isinstance(p, LearnableMask):
                structures_per_pruned_instance = p.non_pruning_size
                max_structures = p.pruning_size
                self.model_sizes_org[n] = (structures_per_pruned_instance, max_structures)

    def print_pruning_stats(self, model):
        """
        print pruning progress
        Args:
            model:

        Returns:

        """
        num_weights_pruned_total = 0
        num_weights_total = 0
        for n, p in model.named_modules():
            if isinstance(p, LearnableMask):
                num_pruned = int((p.p1 * p.lr_mult_factor).int())
                structures_per_pruned_instance = self.model_sizes_org[n][0]  # p.non_pruning_size
                max_structures = self.model_sizes_org[n][1]  # p.pruning_size
                num_pruned = min(num_pruned, max_structures) + self.model_sizes_org[n][1] - p.pruning_size
                num_weights_pruned_total += num_pruned * structures_per_pruned_instance
                num_weights_total += max_structures * structures_per_pruned_instance
                print_log(
                    f"{n}: {num_pruned}/{max_structures} elements pruned; "
                    f"{num_pruned * structures_per_pruned_instance}/{max_structures * structures_per_pruned_instance} weight pruned "
                    f"({100.0 * num_pruned / max_structures}%)",
                    logger='current',
                    level=logging.INFO)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_log("\n_____________________________\n"
                  f"Total weights pruned {self.num_weights_total_first - pytorch_total_params}/{self.num_weights_total_first}"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.INFO)

    def _get_pruning_type(self, module):
        if isinstance(module, torch.nn.Linear):
            return tp.prune_linear_out_channels
        elif isinstance(module, torch.nn.Conv2d):
            return tp.prune_conv_out_channels
        elif isinstance(module, ConvModule):
            return tp.prune_conv_out_channels
        else:
            return tp.prune_multihead_attention_out_channels

    def remove_empty_modules(self, model, module, module_name):
        remove_indexes = self._get_remove_indexes(module)

        if self.debug and len(remove_indexes) >= list(module.masks.values())[0].pruning_size:
            self._remove_module_completely(module_name, model)

    def _remove_module_completely(self, module_name, model):
        print(f"remove completely {module_name} by Empty")
        if "ffn" in module_name:
            module_name = module_name[:module_name.index(".ffn.")+4]
            rsetattr(model, module_name, DummyModule())
            self.history.append(f"deleteffn {module_name}")
            return

        rsetattr(model, module_name, EmptyModule())
        self.history.append(f"delete {module_name}")

    def _reinit_masks(self, model):
        for name, m in model.named_modules():
            if getattr(m, "mask_class_wrapper", False):
                m.reinit_masks()

    def _get_remove_indexes(self, module):
        mask_bias_soft = list(module.masks.values())[0].get_mask()[-1]
        mask_bias = torch.where(mask_bias_soft < 0.001, 1, 0)
        return mask_bias.nonzero(as_tuple=True)[0].tolist()

    def _prune_module(self, model, module, DG, _forward_func, _output_transform):
        remove_indexes = self._get_remove_indexes(module)

        if len(remove_indexes) > 0:
            if isinstance(module, ConvModule):
                module_to_prune = module.conv
            else:
                module_to_prune = module
            tp_type = self._get_pruning_type(module)
            try:
                group = DG.get_pruning_group(module_to_prune, tp_type, idxs=remove_indexes)
            except Exception as e:
                print(e)
                return 0
            if DG.check_pruning_group(group) and len(remove_indexes) > 0:
                model.eval()
                group.prune()
                self._reinit_masks(model)
                return len(remove_indexes)
            else:
                return 0
        else:
            return 0

    """
    Checks weather the given module should be pruned. Return true if yes otherwise false
    """

    def _check_if_prunable(self, module):
        res_bool = True
        # Is a wrapper class
        res_bool &= getattr(module, "mask_class_wrapper", False)
        if not res_bool:
            return False

        first_mask = list(module.masks.values())[0]

        # has at least one mask
        res_bool &= len(list(module.masks.values())) > 0
        #
        is_conv = isinstance(module, torch.nn.Conv2d) or isinstance(module, ConvModule)
        masks_are_convs = isinstance(first_mask, LearnableMaskConv2d)

        is_linear = isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.MultiheadAttention)
        masks_are_linear = isinstance(first_mask, LearnableMaskLinear)

        module_type_bool = (is_conv and masks_are_convs) or (is_linear and masks_are_linear)

        return res_bool and module_type_bool

    def prune_weight(self, model):
        if hasattr(model, 'data_preprocessor'):
            data_batch = model.data_preprocessor(self.get_data_batch((3, 512, 512)))
        else:
            data_batch = model.module.data_preprocessor(self.get_data_batch((3, 512, 512)))
        set_identity_layer_mode(model, True)

        def _forward_func(self, data):
            return self.test_step(data)

        def _output_transform(data):
            l = []
            for i in data:
                l.append(i.seg_logits.data)
            return l

        num_removed = 0
        # Remove all completely empty modules
        for module_name, module in model.named_modules():
            if self._check_if_prunable(module):
                self.remove_empty_modules(model, module, module_name)

        # get dependency grapgh for pruning
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=data_batch, forward_fn=_forward_func,
                                                   output_transform=_output_transform)

        # prune modules
        for module_name, module in model.named_modules():
            if self._check_if_prunable(module):
                num_removed += self._prune_module(model, module, DG, _forward_func, _output_transform)

        self.history.extend(DG.pruning_history())
        set_identity_layer_mode(model, False)
        self.remove_total += num_removed
        print_log("\n_____________________________\n"
                  f"Total weights removed {self.remove_total} (+ {num_removed})"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.DEBUG)

    def save_history(self, runner):
        history_filename = osp.join(runner.work_dir, f"iter_{runner.iter + 1}.history")
        history_filename_last = osp.join(runner.work_dir, f"last_history")
        with open(history_filename, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(history_filename_last, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:

        if self.num_weights_total_first == -1:
            pytorch_total_params = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
            self.num_weights_total_first = pytorch_total_params
            self.init_model_stats(runner.model)
            print_log("\n_____________________________\n"
                      f"Total weights of model {self.num_weights_total_first}"
                      "\n_____________________________\n",
                      logger='current',
                      level=logging.INFO)

        if (self.prune_at_start and runner.iter == 50) or (self.do_explicit_pruning and (
                self.every_n_train_iters(runner, self.pruning_interval) or self.is_last_train_iter(runner))):
            self.prune_weight(runner.model)

        if self.logging and (
                self.every_n_train_iters(runner, self.logging_interval) or self.is_last_train_iter(runner)):
            self.print_pruning_stats(runner.model)

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        self.save_history(runner)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        pass
