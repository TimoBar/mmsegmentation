# Copyright (c) OpenMMLab. All rights reserved.
import torch
import argparse
import logging
import os
import os.path as osp
from mmengine.config import Config, DictAction




"""
Since I have to overwrite the classes used in the models, the order of the import is important
"""

def overwrite_classes_static_pruning():
    from mmseg.models.utils.static_pruning import StaticMaskConv2d, StaticMaskLinear, mask_class_wrapper

    torch.nn.Conv2d_base = torch.nn.Conv2d
    torch.nn.Conv2d = mask_class_wrapper(torch.nn.Conv2d_base, mode="conv")

    torch.nn.MultiheadAttention_base = torch.nn.MultiheadAttention
    torch.nn.MultiheadAttention = mask_class_wrapper(torch.nn.MultiheadAttention_base, mode="linear")
    torch.nn.Linear_base = torch.nn.Linear
    torch.nn.Linear = mask_class_wrapper(torch.nn.Linear_base, mode="linear")

    from mmseg.registry import MODELS
    import mmcv
    mmcv.cnn.Conv2d_base = mmcv.cnn.Conv2d
    mmcv.cnn.Conv2d = mask_class_wrapper(mmcv.cnn.Conv2d_base, mode="conv")
    MODELS.register_module('Conv2d', module=torch.nn.Conv2d)
def overwrite_classes_logistic_pruning():
    from mmseg.models.utils import mask_class_wrapper

    torch.nn.Conv2d_base = torch.nn.Conv2d
    torch.nn.Conv2d = mask_class_wrapper(torch.nn.Conv2d_base, mode="conv")

    torch.nn.MultiheadAttention_base = torch.nn.MultiheadAttention
    torch.nn.MultiheadAttention = mask_class_wrapper(torch.nn.MultiheadAttention_base, mode="linear")
    torch.nn.Linear_base = torch.nn.Linear
    torch.nn.Linear = mask_class_wrapper(torch.nn.Linear_base, mode="linear")

    from mmseg.registry import MODELS
    import mmcv
    mmcv.cnn.Conv2d_base = mmcv.cnn.Conv2d
    mmcv.cnn.Conv2d = mask_class_wrapper(mmcv.cnn.Conv2d_base, mode="conv")
    MODELS.register_module('Conv2d', module=torch.nn.Conv2d)

def overwrite_classes_logistic_kernel_pruning():
    from mmseg.models.utils import mask_class_wrapper

    torch.nn.Conv2d_base = torch.nn.Conv2d
    torch.nn.Conv2d = mask_class_wrapper(torch.nn.Conv2d_base, mode="kernel")

    torch.nn.MultiheadAttention_base = torch.nn.MultiheadAttention
    torch.nn.MultiheadAttention = mask_class_wrapper(torch.nn.MultiheadAttention_base, mode="linear")
    torch.nn.Linear_base = torch.nn.Linear
    torch.nn.Linear = mask_class_wrapper(torch.nn.Linear_base, mode="linear")

    from mmseg.registry import MODELS
    import mmcv
    mmcv.cnn.Conv2d_base = mmcv.cnn.Conv2d
    mmcv.cnn.Conv2d = mask_class_wrapper(mmcv.cnn.Conv2d_base, mode="kernel")
    MODELS.register_module('Conv2d', module=torch.nn.Conv2d)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--explicit-pruning',
        action='store_true',
        default=False,
        help='Prunes the weight not just by setting them to zero, but by removing the corresponding row out of the parameter tensor')
    parser.add_argument(
        '--pruning-mode',
        default="logistic",
        help='Pruning method: Can be "static", "logistic"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    if args.pruning_mode == "logistic":
        overwrite_classes_logistic_pruning()
    elif args.pruning_mode == "logistic_kernel":
        overwrite_classes_logistic_kernel_pruning()
    elif args.pruning_mode == "static":
        overwrite_classes_static_pruning()
    elif args.pruning_mode == "acosp":
        pass
    else:
        raise NotImplementedError(f"The pruning mode {args.pruning_mode} is not yet implemented")

    from mmengine.logging import print_log
    from mmengine.runner import Runner

    from mmseg.registry import RUNNERS

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    if args.pruning_mode == "acosp":
        cfg["custom_hooks"] = [dict(type='AcospHook', interval=2975, max_iters=cfg.train_cfg.max_iters)]
    elif args.pruning_mode == "logistic" or args.pruning_mode == "logistic_kernel":
        cfg["custom_hooks"] = [dict(type='LogisticWeightPruningHook', do_explicit_pruning=args.explicit_pruning, logging_interval=5000, pruning_interval=10)]

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)



    # start training
    runner.train()


if __name__ == '__main__':
    main()
