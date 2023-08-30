import functools
import os
import pickle
from typing import Any, Optional, Union

import mmcv
import torch
from mmcv.cnn import ConvModule
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from torch.nn.modules.module import register_module_forward_pre_hook

from mmseg.models.backbones.mit_prunable import EfficientMultiheadAttention_Conv2d_pruned, MixFFN_Conv2d_pruned
from mmseg.models.utils import reduce_feature_dim, set_identity_layer_mode
from mmseg.models.utils.identity_conv import EmptyModule
from mmseg.models.utils.mask_wrapper import LearnableMask, LearnableKernelMask, LearnableMaskLinear, \
    LearnableMaskConv2d, rgetattr, LearnableMaskMHA, rsetattr
import logging
from mmengine.logging import print_log
import torch_pruning as tp
import os.path as osp

import traceback
import sys

from mmseg.structures import SegDataSample

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class LogisticWeightPruningHook(Hook):
    """
    """
    priority = 'LOW'
    def __init__(self, do_logging=True, do_explicit_pruning=True, logging_interval=100, pruning_interval=25, debug=False):
        self.logging_interval = logging_interval
        self.pruning_interval = pruning_interval
        self.logging = do_logging
        self.do_explicit_pruning = do_explicit_pruning
        self.remove_total = 0
        self.num_weights_total_first = -1
        self.model_sizes_org = {}
        self.history = []
        self.debug = debug

    def find_last_history(self, path):
        save_file = osp.join(path, 'last_history')
        if os.path.exists(save_file):
            return save_file
        else:
            print_log('Did not find last_history to be resumed.')
            return None

    def get_data_batch(self, input_shape):
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
        return runner._resume or runner._load_from

    def before_run(self, runner) -> None:
        history_filename = self.find_last_history(runner.work_dir)
        if history_filename is not None and self.is_resume(runner):
            #print(f"load {history_filename}")
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
            if isinstance(p, LearnableKernelMask):
                pass
            elif isinstance(p, LearnableMask):
                structures_per_pruned_instance = p.non_pruning_size
                max_structures = p.pruning_size
                self.model_sizes_org[n] = (structures_per_pruned_instance, max_structures)


    def print_pruning_stats(self, model):
        num_weights_pruned_total = 0
        num_weights_total = 0
        for n, p in model.named_modules():
            if isinstance(p, LearnableKernelMask):
                p1 = torch.min(torch.max(torch.zeros_like(p.p1).int(), (p.p1 * p.factor).int()),
                               torch.ones_like(p.p1) * p.kernel_size)
                p1_x = p1[0:p1.size(0) // 2]
                p1_y = p1[p1.size(0) // 2:]
                his = torch.histogramdd(torch.stack([p1_x, p1_y]).permute(1, 0).cpu(),
                                        bins=[p.kernel_size + 1, p.kernel_size + 1],
                                        range=[0, p.kernel_size, 0, p.kernel_size])

                num_pruned_completely = int(torch.sum(his.hist[-1, :]) + torch.sum(his.hist[0:-1, -1]))
                num_weights = p.input_features * p.output_features * p.kernel_size * p.kernel_size
                num_weights_total += num_weights

                sizes = {}
                for i in range(p.output_features):
                    kernel_size = int((p.kernel_size - p1_x[i]) * (p.kernel_size - p1_y[i]))
                    sizes[kernel_size] = (sizes[kernel_size] + 1) if kernel_size in sizes else 1
                #print(sizes)
                num_weights_not_pruned = 0
                for kernel_size, num in sizes.items():
                    num_weights_not_pruned += p.input_features * float(kernel_size) * num

                num_weights_pruned = num_weights - int(num_weights_not_pruned)
                num_weights_pruned_total += num_weights_pruned
                print_log(f"{n}: Kernels:\n{his.hist[0:-1, 0:-1]}\n "
                          f"Num removed kernels {num_pruned_completely}, "
                          f"Num weights pruned {num_weights_pruned}/{num_weights} ({100.0 * num_weights_pruned/num_weights}%)",
                          logger='current',
                          level=logging.INFO)

            elif isinstance(p, LearnableMask):
                num_pruned = int((p.p1 * p.factor).int())
                structures_per_pruned_instance = self.model_sizes_org[n][0]#p.non_pruning_size
                max_structures = self.model_sizes_org[n][1]#p.pruning_size
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
                  f"Total weights pruned {self.num_weights_total_first- pytorch_total_params}/{self.num_weights_total_first}"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.INFO)

    def prune_weight(self, model):
        data_batch = model.module.data_preprocessor(self.get_data_batch((3, 512, 512)))
        set_identity_layer_mode(model, True)
        def _forward_func(self, data):
            return self.test_step(data)

        def _output_transform(data):
            l = []
            for i in data:
                l.append(i.seg_logits.data)
            return l

        #model.eval()
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=data_batch, forward_fn=_forward_func, output_transform=_output_transform)
        #model.train()
        num_removed = 0
        for module_name, module in model.named_modules():
            if getattr(module, "mask_class_wrapper", False):
                if len(list(module.masks.values())) > 0 and (
                        ((isinstance(module, torch.nn.Conv2d) or isinstance(module, ConvModule))
                         and (isinstance(list(module.masks.values())[0], LearnableMaskConv2d) or isinstance(
                                    list(module.masks.values())[0], LearnableMaskMHA))) \
                        or ((isinstance(module, torch.nn.Linear) or isinstance(module,
                                                                               torch.nn.MultiheadAttention)) and isinstance(
                    list(module.masks.values())[0], LearnableMaskLinear))):

                    #if self.debug and ("decode_head.convs" in module_name):
                    #    print(f"Skipped module {module_name}")
                    #    continue
                    mask_bias_soft = list(module.masks.values())[0].get_mask()[-1]

                    mask_bias = torch.where(mask_bias_soft < 0.001, 1, 0)
                    remove_indexes = mask_bias.nonzero(as_tuple=True)[0].tolist()

                    if len(remove_indexes) > 0:
                        mo = module
                        if isinstance(module, torch.nn.Linear):
                            tp_type = tp.prune_linear_out_channels
                        elif isinstance(module, torch.nn.Conv2d):
                            tp_type = tp.prune_conv_out_channels
                        elif isinstance(module, ConvModule):
                            mo = module.conv
                            tp_type = tp.prune_conv_out_channels
                        else:
                            tp_type = tp.prune_multihead_attention_out_channels

                        if self.debug and len(remove_indexes) >= list(module.masks.values())[0].pruning_size:
                            print(f"remove completely {module_name} by Empty")
                            rsetattr(model, module_name, EmptyModule())
                            data_batch = model.module.data_preprocessor(self.get_data_batch((3, 512, 512)))
                            self.history.append(f"delete {module_name}")
                            DG = tp.DependencyGraph().build_dependency(model, example_inputs=data_batch,
                                                                       forward_fn=_forward_func,
                                                                       output_transform=_output_transform)
                            continue

                        group = DG.get_pruning_group(mo, tp_type, idxs=remove_indexes)
                        if not DG.check_pruning_group(group):

                            mask_object = list(module.masks.values())[0]
                            with torch.no_grad():
                                mask_object.p1.copy_(mask_object.p1 - 1 / mask_object.factor)
                            mask_bias = torch.where(mask_bias < 0.0001, 1, 0)
                            remove_indexes = mask_bias.nonzero(as_tuple=True)[0].tolist()
                            group = DG.get_pruning_group(mo, tp_type, idxs=remove_indexes)
                        # print(group)
                        if DG.check_pruning_group(group) and len(remove_indexes) > 0:
                            num_removed += len(remove_indexes)
                            model.eval()
                            #with torch.no_grad():
                            #    before = _forward_func(model, data_batch)[0].seg_logits.data.cpu()

                            group.prune()

                            for name, m in model.named_modules():
                                if getattr(m, "mask_class_wrapper", False):
                                    m.reinit_masks()

                            # model.eval()
                            with torch.no_grad():
                                after = _forward_func(model, data_batch)[0].seg_logits.data.cpu()

                            #diff = torch.sum(torch.abs(before - after)) / torch.numel(before)
                            #print(f"Difference after pruning: {diff}")
                            """DG = tp.DependencyGraph().build_dependency(model, example_inputs=data_batch,
                                                                       forward_fn=_forward_func,
                                                                       output_transform=_output_transform)"""
                        else:
                            pass
                            # print("Cant prune this layer")

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

        if self.do_explicit_pruning and (
                self.every_n_train_iters(runner, self.pruning_interval) or self.is_last_train_iter(runner)):
            self.prune_weight(runner.model)

        if self.logging and (
                self.every_n_train_iters(runner, self.logging_interval) or self.is_last_train_iter(runner)):
            self.print_pruning_stats(runner.model)

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        self.save_history(runner)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        pass
