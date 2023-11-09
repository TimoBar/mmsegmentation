# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmengine import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.runner import Runner
from torch import nn
import os.path as osp

from mmseg.apis import inference_model, init_model, show_result_pyplot
from mmseg.engine.hooks import PruningHook


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    print("Create Model")
    config = Config.fromfile(args.config)
    config.model.backbone.k = 7
    config.model.decode_head.k = 7
    model = init_model(config, args.checkpoint, device=args.device)
    hook = PruningHook(do_explicit_pruning=True,
                       logging_interval=5000, pruning_interval=5000, debug=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hook.num_weights_total_first = pytorch_total_params
    hook.init_model_stats(model)
    hook.prune_weight(model)
    hook.print_pruning_stats(model)

    print("Model created")
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # test a single image
    print("Start Inference")
    result = inference_model(model, args.img)
    print("Inference finished")
    # show the results
    print(result)
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
