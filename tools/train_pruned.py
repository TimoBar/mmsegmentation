# Copyright (c) OpenMMLab. All rights reserved.
import sys, os
if '/beegfs/work/bartels/mmsegmentation' in sys.path:
    sys.path.remove('/beegfs/work/bartels/mmsegmentation')

sys.path.append(os.getcwd())


import torch
import argparse
import logging
import os
import os.path as osp
from mmengine.config import Config, DictAction
from mmengine.runner import find_latest_checkpoint
from torch import autograd

from tools.test_pruned import soft2hard_prunung


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
        '--load-from',
        type=str,
        default=None,
        help='Checkpoint to load init weights')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--pruning-interval',
        type=int,
        default=5000,
        help='interval of hard pruning of model weights')
    parser.add_argument(
        '--pruning-logging-interval',
        type=int,
        default=5000,
        help='interval of logging of soft pruning of model weights')
    parser.add_argument(
        '--with-flops',
        action='store_true',
        default=False,
        help='if flops should be measured during training')
    parser.add_argument(
        '--with-fps',
        action='store_true',
        default=False,
        help='if fps should be measured during training')
    parser.add_argument(
        '--flops-interval',
        type=int,
        default=5000,
        help='interval of measuring flops of model')
    parser.add_argument(
        '--flops-input-size',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='the shape of the input that is used to compute the flops')
    parser.add_argument(
        '--fps-interval',
        type=int,
        default=5000,
        help='interval of measuring fps of model')
    parser.add_argument(
        '--explicit-pruning',
        action='store_true',
        default=False,
        help='Prunes the weight not just by setting them to zero, but by removing the corresponding row out of the parameter tensor')
    parser.add_argument(
        '--prune-at-start',
        action='store_true',
        default=False,
        help='when true the model is pruned in first training iteration')
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
    if not args.explicit_pruning:
        torch.backends.cudnn.benchmark = True

    if args.pruning_mode == "segformer":
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
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

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

    if args.load_from is not None:
        resume_from = None
        if args.resume:
            resume_from = find_latest_checkpoint(cfg.work_dir)

        if resume_from is None:
            cfg.load_from = args.load_from
            cfg.resume = False

    hooks = []
    if args.pruning_mode == "segformer":
        hooks.append(dict(type='PruningHook', do_explicit_pruning=args.explicit_pruning, prune_at_start=args.prune_at_start, logging_interval=args.pruning_logging_interval, pruning_interval=args.pruning_interval, debug=True))

    if args.with_fps:
        hooks.append(dict(type='FPSMeasureHook', interval=args.fps_interval))

    if args.with_flops:
        if len(args.flops_input_size) != 2:
            raise Exception(f"You have to give exact 2 numbers for the argument --flops-input-size, but {len(args.flops_input_size)} were given")
        hooks.append(dict(type='FLOPSMeasureHook', model_cfg=cfg["model"], interval=args.flops_interval, input_shape=tuple(args.flops_input_size)))

    cfg["custom_hooks"] = hooks

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    #torch.autograd.set_detect_anomaly(True)
    # start training
    runner.train()


if __name__ == '__main__':
    main()
