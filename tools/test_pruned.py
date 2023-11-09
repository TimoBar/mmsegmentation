# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmseg.engine.hooks import PruningHook, FLOPSMeasureHook, FPSMeasureHook

from zeus.monitor import ZeusMonitor


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualizer = cfg.visualizer
            visualizer['save_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg
def print_fps(runner):
    for hook in runner._hooks:
        if isinstance(hook, FPSMeasureHook):
            hook.before_train(runner)
def soft2hard_prunung(runner):
    for hook in runner._hooks:
        if isinstance(hook, PruningHook):
            pytorch_total_params = sum(p.numel() for p in runner.model.parameters() if p.requires_grad)
            hook.num_weights_total_first = pytorch_total_params
            hook.init_model_stats(runner.model)
            hook.prune_weight(runner.model)
            hook.print_pruning_stats(runner.model)

    for hook in runner._hooks:
        if isinstance(hook, FLOPSMeasureHook):
            hook.print_flops(hook.measure_flops(runner))

def test_loop_run(loop):
    loop.runner.call_hook('before_test')
    loop.runner.call_hook('before_test_epoch')
    loop.runner.model.eval()
    max_iters = 100000
    for idx, data_batch in enumerate(loop.dataloader):
        loop.run_iter(idx, data_batch)
        if idx >= max_iters:
            break

    # compute metrics
    metrics = loop.evaluator.evaluate(max_iters)
    #metrics = loop.evaluator.evaluate(len(loop.dataloader.dataset))
    loop.runner.call_hook('after_test_epoch', metrics=metrics)
    loop.runner.call_hook('after_test')
    return metrics
def runner_test(runner):
    if runner._test_loop is None:
        raise RuntimeError(
            '`self._test_loop` should not be None when calling test '
            'method. Please provide `test_dataloader`, `test_cfg` and '
            '`test_evaluator` arguments when initializing runner.')

    #print_fps(runner)
    runner._test_loop = runner.build_test_loop(runner._test_loop)  # type: ignore

    runner.call_hook('before_run')

    # make sure checkpoint-related hooks are triggered after `before_run`
    runner.load_or_resume()

    soft2hard_prunung(runner)
    #print_fps(runner)

    monitor = ZeusMonitor(gpu_indices=[0])
    monitor.begin_window("heavy computation")
    metrics = test_loop_run(runner.test_loop)  # type: ignore
    runner.call_hook('after_run')

    measurement = monitor.end_window("heavy computation")
    print(f"Energy: {measurement.total_energy} J")
    print(f"Time  : {measurement.time} s")

    return metrics

def main():
    args = parse_args()

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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    hooks = []
    cfg.model.backbone.pretrained = None
    cfg.model.pretrained = None
    method = "own"#"dynasegformer"#"own"
    if method == "own":
        hooks.append(dict(type='PruningHook', do_explicit_pruning=True,
                          logging_interval=5000, pruning_interval=5000, debug=True))
        hooks.append(dict(type='FLOPSMeasureHook', model_cfg=cfg["model"], interval=5000,
                          input_shape=(2048, 1024)))
        hooks.append(dict(type='FPSMeasureHook', interval=5000))
        cfg["custom_hooks"] = hooks

        cfg.model.backbone.k = 7
        cfg.model.decode_head.k = 7

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing

    runner_test(runner)

if __name__ == '__main__':
    main()
