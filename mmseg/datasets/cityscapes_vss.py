# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List, Any, Sequence, Optional, Union, Dict

import mmengine
import numpy as np
from mmengine import fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from .cityscapes import CityscapesDataset
import os.path as osp
import copy
import pickle


@DATASETS.register_module()
class CityscapesVideoDataset(CityscapesDataset):
    CLASSES = None

    def __init__(self,
                 seg_map_suffix='_gtFine_labelIds.png',
                 key_frame_index=19,
                 frames_before=6,
                 frames_after=0,
                 test_load_ann=False,
                 *args,
                 **kwargs):

        self.test_load_ann = test_load_ann
        self.key_frame_index = key_frame_index
        self.frames_before = frames_before
        self.frames_after = frames_after
        super().__init__(seg_map_suffix=seg_map_suffix,*args, **kwargs)
        # self.logger = get_root_logger()

    def find(self, str, ch):
        for i, ltr in enumerate(str):
            if ltr == ch:
                yield i

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        """if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:"""
        _suffix_len = len(self.img_suffix)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            data_info = dict(img_path=osp.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img[:-_suffix_len] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)

        # group images by their corresponding sequence
        d = {}
        for data in data_list:
            key_end_index = list(self.find(data["img_path"], "_"))[-2]
            key = data["img_path"][0:key_end_index]
            if key not in d:
                d[key] = [data]
            else:
                d[key].append(data)
                d[key] = sorted(d[key], key=lambda x: x['img_path'])

        # sort inner list
        for data in data_list:
            key_end_index = list(self.find(data["img_path"], "_"))[-2]
            key = data["img_path"][0:key_end_index]
            d[key] = sorted(d[key], key=lambda x: x['img_path'])

        data_list = list(d.values())
        additional_data = []
        sequence_length = 30
        for i in range(len(data_list) - 1, -1, -1):
            l = len(data_list[i])
            if l > sequence_length:
                for j in range(0, l // sequence_length, 1):
                    additional_data.append(data_list[i][j * sequence_length:(j + 1) * sequence_length])
                data_list.pop(i)

        data_list.extend(additional_data)

        outs = []
        for inner_list in data_list:
            outs.append(inner_list[self.key_frame_index - self.frames_before:self.key_frame_index + self.frames_after + 1])
        data_list = outs
        # sort outer list
        data_list = sorted(data_list, key=lambda x: x[0]['img_path'])

        """#log
        for data in data_list:
            for sample_data in data:
                print(sample_data['img_path'])
            print("________________________________________________________________________________")"""
        return data_list

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])

        return self.pipeline(data_info)

    def get_label_map(cls,
                      new_classes: Optional[Sequence] = None
                      ) -> Union[Dict, None]:
        return {
            0: 255,
            1: 255,
            2: 255,
            3: 255,
            4: 255,
            5: 255,
            6: 255,
            7: 0,
            8: 1,
            9: 255,
            10: 255,
            11: 2,
            12: 3,
            13: 4,
            14: 255,
            15: 255,
            16: 255,
            17: 5,
            18: 255,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: 255,
            30: 255,
            31: 16,
            32: 17,
            33: 18,
            -1: 255
        }
