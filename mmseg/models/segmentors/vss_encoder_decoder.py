# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import nullcontext
from typing import List, Optional, Union, Dict

from mmengine.optim import OptimWrapper
from mmengine.runner.loops import IterBasedTrainLoop

import torch
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from torch import Tensor, nn

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, ForwardResults)
from .encoder_decoder import EncoderDecoder
from ..utils import ConvLSTM


@MODELS.register_module()
class VSSEncoderDecoder(EncoderDecoder):
    """Video Semantic Segmentation Encoder Decoder segmentors.


    """

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 use_clstm_neck=True,
                 predict_only_labeled=False,
                 freeze_backbone=False,
                 freeze_decoder_head=False,
                 teacher: ConfigType = None,
                 teacher_checkpoint: str = None,
                 key_frame_idx: int = -1,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.use_clstm_neck = use_clstm_neck
        if self.use_clstm_neck:
            self.clstm = ConvLSTM(img_size=(32, 32), input_dim=256, hidden_dim=256, kernel_size=(3, 3), cnn_dropout=0.2,
                                  rnn_dropout=0.2, batch_first=False, bias=False)
            self.clstm_hidden_state = None

        self.key_frame_idx = key_frame_idx
        self.use_clstm_neck = use_clstm_neck
        self.predict_only_labeled = predict_only_labeled
        self.freeze_backbone = freeze_backbone
        self.freeze_decoder_head = freeze_decoder_head


        self.teacher = teacher
        if self.teacher is not None:
            self.teacher = MODELS.build(teacher)
            checkpoint = _load_checkpoint(teacher_checkpoint, map_location='cpu')
            _load_checkpoint_to_model(self.teacher, checkpoint)
            self.teacher = self.teacher.cuda()

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        x = self.extract_feat(inputs)
        # print(torch.unsqueeze(x[-1], 0).size(), self.clstm_hidden_state)
        if self.use_clstm_neck:
            x_mod, self.clstm_hidden_state, _ = self.clstm(torch.unsqueeze(x[-1], 0), self.clstm_hidden_state)
            x.pop(-1)
            x.append(x_mod[:, 0, :, :, :])

        with torch.no_grad() if self.freeze_decoder_head else nullcontext():
            seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    """def _decode_head_forward_train(self, inputs: Tensor,
                                   data_samples: SampleList) -> dict:
        return super()._decode_head_forward_train(inputs=inputs, data_samples=data_samples)"""

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        x = self.extract_feat(inputs)
        if self.use_clstm_neck:
            x_mod, self.clstm_hidden_state, _ = self.clstm(torch.unsqueeze(x[-1], 0), self.clstm_hidden_state)
            x.pop(-1)
            x.append(x_mod[:, 0, :, :, :])
        return self.decode_head.forward(x)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            x = self.backbone(inputs)

        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, inputs: List[Tensor], data_samples: List[SampleList]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # Encoder forward
        x_all = []
        for inputs_t, data_samples_t in zip(inputs, data_samples):
            x_all.append(self.extract_feat(inputs_t))

        # CLSTM Forward
        if self.use_clstm_neck:
            last = []
            for x in x_all:
                last.append(x[-1])
            r = self.clstm(torch.stack(last))[0]
            x_all_mod = []
            for i, x in enumerate(x_all):
                t = x[0:-1]
                t.append(r[:, i, :, :, :])
                x_all_mod.append(t)
        else:
            x_all_mod = x_all

        # decoder_head forward
        outs = []
        num_samples = 0
        for x, data_samples_t in zip(x_all_mod, data_samples):
            # Skip loss calculation for frames with no ground truth
            # Only applies when no teacher is used
            if '_gt_sem_seg' not in data_samples_t[0]:
                continue
            losses = dict()

            with torch.no_grad() if self.freeze_decoder_head else nullcontext():
                loss_decode = self._decode_head_forward_train(x, data_samples_t)
            losses.update(loss_decode)

            if self.with_auxiliary_head:
                loss_aux = self._auxiliary_head_forward_train(x, data_samples_t)
                losses.update(loss_aux)

            if not outs:
                outs = losses
            else:
                outs = self.add_dict(outs, losses)

                # parsed_losses, log_vars = self.parse_losses(outs)
                # parsed_losses.backward()
                # outs = self.loss(inputs_t, data_samples_t)
            num_samples += 1
        outs = self.divide_dict(outs, num_samples)
        return outs

    def add_dict(self, a, b):
        for key in a.keys():
            a[key] = a[key] + b[key]
        return a

    def divide_dict(self, a, x):
        for key in a.keys():
            a[key] = a[key] / x
        return a

    def forward(self,
                inputs: List[Tensor],
                data_samples: List[OptSampleList] = None,
                mode: str = 'tensor') -> ForwardResults:

        if mode == 'loss':
            if self.teacher is not None:
                # Teacher predictions
                for inputs_t, data_samples_t in zip(inputs, data_samples):
                    data_samples_t[0].keys()
                    if '_gt_sem_seg' not in data_samples_t[0]:
                        with torch.no_grad():
                            seg_data_sample = self.teacher.predict(inputs_t, None)
                            for a, b in zip(data_samples_t, seg_data_sample):
                                a.set_field(name="_gt_sem_seg", value=b.get("pred_sem_seg"))
            return self.loss(inputs, data_samples)
        else:
            outs = []
            for i, inputs_t in enumerate(inputs):
                if self.predict_only_labeled and i != self.key_frame_idx and i != len(inputs) + self.key_frame_idx:
                    outs.append(None)
                    continue

                if mode == 'predict':
                    outs.append(self.predict(inputs_t, data_samples[self.key_frame_idx]))
                elif mode == 'tensor':
                    outs.append(self._forward(inputs_t, data_samples[self.key_frame_idx]))
                else:
                    raise RuntimeError(f'Invalid mode "{mode}". '
                                       'Only supports loss, predict and tensor mode')
            # reset hidden state
            self.clstm_hidden_state = None
            #print(outs)
            #print(outs[self.key_frame_idx])
            return outs[self.key_frame_idx]
