# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
import os.path as osp
import numpy as np

from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS, VISUALIZERS
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    register_all_modules()

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = ProgressBar(len(dataset))
    counted_classes_sum = 0
    for item in dataset:
        #mask = item['targets']
        #img = img[..., [2, 1, 0]]  # bgr to rgb
        data_sample = torch.flatten(item['data_samples'].gt_sem_seg.data)
        data_sample = torch.where(data_sample == 255, torch.ones_like(data_sample) * 150, data_sample)
        counted_classes = torch.bincount(data_sample, minlength=151)
        counted_classes_sum += counted_classes
        #print(counted_classes_sum)
        #img_path = osp.basename(item['data_samples'].img_path)


        progress_bar.update()

    print(counted_classes_sum.numpy())
    np.savetxt("ade20k.txt", counted_classes_sum.numpy())

if __name__ == '__main__':
    main()
