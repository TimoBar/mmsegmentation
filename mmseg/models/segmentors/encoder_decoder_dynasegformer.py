# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional

import mmcv
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import is_model_wrapper
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model, load_state_dict
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from mmcv.cnn import Conv2d
from .encoder_decoder import EncoderDecoder
from ..utils import mask_class_wrapper, get_num_pruned, get_p1_values, get_p1_loss, get_dgl_layer_loss



@MODELS.register_module()
class EncoderDecoderDynaSegFormer(EncoderDecoder):

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 mask_regularizer=0.005,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 teacher=None,
                 teacher_checkpoint=None,
                 lamda_s=0.5):
        self.mask_regularizer = mask_regularizer
        super().__init__(backbone=backbone, decode_head=decode_head, neck=neck, auxiliary_head=auxiliary_head,
                         train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained,
                         data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if teacher is not None:
            if isinstance(teacher, dict) and teacher["data_preprocessor"] is None:
                # Merge the data_preprocessor to model config.
                teacher.setdefault('data_preprocessor', data_preprocessor)
            self.teacher = MODELS.build(teacher)
            self.soft_ce_loss = torch.nn.CrossEntropyLoss()
            self.mse_loss = nn.MSELoss()
            self.lamda_s = lamda_s

            if is_model_wrapper(self.teacher):
                self.teacher = self.teacher.module
            else:
                self.teacher = self.teacher
            checkpoint = _load_checkpoint(teacher_checkpoint, map_location='cpu')
            """if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict._metadata = getattr(state_dict, '_metadata', OrderedDict())
            load_state_dict(self.teacher, state_dict, False, None)"""
            checkpoint = _load_checkpoint_to_model(self.teacher, checkpoint)

            self.teacher = self.teacher.cuda()

        else:
            self.teacher = None
        #self.teacher = build_segmentor(teacher)

    def _get_teacher_predictions(self, inputs: Tensor):
        with torch.no_grad():
            x_teacher = self.teacher.extract_feat(inputs)
            logits_teacher = self.teacher.decode_head.forward(x_teacher)

        return x_teacher, logits_teacher

    def _get_knowledge_distillation_losses(self, x_student, x_teacher, logits_student, logits_teacher):
        stage_1_kd_loss = 0
        for xi_student, xi_teacher in zip(x_student, x_teacher):
            stage_1_kd_loss = stage_1_kd_loss + self.mse_loss(xi_student, xi_teacher)

        stage_2_kd_loss = self.soft_ce_loss(logits_student, F.softmax(logits_teacher, dim=1)) * self.lamda_s

        return stage_1_kd_loss, stage_2_kd_loss

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)

        losses = dict()
        seg_logits = self.forward(inputs)
        loss_decode = self.decode_head.loss_by_feat(seg_logits, data_samples)

        losses.update(add_prefix(loss_decode, 'decode'))
        losses.update(loss_decode)

        mask_loss = get_dgl_layer_loss(self) * self.mask_regularizer
        losses["decode.mask_loss"] = mask_loss
        losses["decode.loss_ce"] = losses["decode.loss_ce"]

        if self.teacher is not None:
            x_teacher, logits_teacher = self._get_teacher_predictions(inputs)
            stage_1_kd_loss, stage_2_kd_loss = self._get_knowledge_distillation_losses(x, x_teacher, seg_logits, logits_teacher)
            losses["decode.stage_1_kd_loss"] = stage_1_kd_loss
            losses["decode.stage_2_kd_loss"] = stage_2_kd_loss
        #losses.update(add_prefix(get_num_pruned(self), 'decode.pruned'))

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses