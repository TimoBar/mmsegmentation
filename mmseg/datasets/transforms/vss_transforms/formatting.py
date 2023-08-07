# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import cv2
import numpy as np

from ..formatting import PackSegInputs
from mmseg.registry import TRANSFORMS
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform

from mmengine.structures import PixelData
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class ConcatSameTypeFrames(object):
    """Concat the frames of the same type. We divide all the frames into two
    types: 'key' frames and 'reference' frames.

    The input list contains as least two dicts. We concat the first
    `num_key_frames` dicts to one dict, and the rest of dicts are concated
    to another dict.

    In SOT field, 'key' denotes template image and 'reference' denotes search
    image.

    Args:
        num_key_frames (int, optional): the number of key frames.
            Defaults to 1.
    """

    def __init__(self, num_key_frames=1):
        self.num_key_frames = num_key_frames

    def concat_one_mode_results(self, results):
        """Concatenate the results of the same mode."""
        out = dict()
        for i, result in enumerate(results):
            if 'img' in result:
                img = result['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if i == 0:
                    result['img'] = np.expand_dims(img, -1)
                else:
                    out['img'] = np.concatenate(
                        (out['img'], np.expand_dims(img, -1)), axis=-1)
            for key in ['img_metas', 'gt_masks']:
                if key in result:
                    if i == 0:
                        result[key] = [result[key]]
                    else:
                        out[key].append(result[key])
            for key in [
                    'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                    'gt_instance_ids'
            ]:
                if key not in result:
                    continue
                value = result[key]
                if value.ndim == 1:
                    value = value[:, None]
                N = value.shape[0]
                value = np.concatenate((np.full(
                    (N, 1), i, dtype=np.float32), value),
                                       axis=1)
                if i == 0:
                    result[key] = value
                else:
                    out[key] = np.concatenate((out[key], value), axis=0)
            if 'gt_semantic_seg' in result:
                if i == 0:
                    result['gt_semantic_seg'] = result['gt_semantic_seg'][...,
                                                                          None,
                                                                          None]
                else:
                    out['gt_semantic_seg'] = np.concatenate(
                        (out['gt_semantic_seg'],
                         result['gt_semantic_seg'][..., None, None]),
                        axis=-1)

            if 'padding_mask' in result:
                if i == 0:
                    result['padding_mask'] = np.expand_dims(
                        result['padding_mask'], 0)
                else:
                    out['padding_mask'] = np.concatenate(
                        (out['padding_mask'],
                         np.expand_dims(result['padding_mask'], 0)),
                        axis=0)

            if i == 0:
                out = result
        return out

    def __call__(self, results):
        """Call function.

        Args:
            results (list[dict]): list of dict that contain keys such as 'img',
                'img_metas', 'gt_masks','proposals', 'gt_bboxes',
                'gt_bboxes_ignore', 'gt_labels','gt_semantic_seg',
                'gt_instance_ids', 'padding_mask'.

        Returns:
            list[dict]: The first dict of outputs concats the dicts of 'key'
                information. The second dict of outputs concats the dicts of
                'reference' information.
        """
        assert (isinstance(results, list)), 'results must be list'
        key_results = []
        reference_results = []
        for i, result in enumerate(results):
            if i < self.num_key_frames:
                key_results.append(result)
            else:
                reference_results.append(result)
        outs = []
        if self.num_key_frames == 1:
            # if single key, not expand the dim of variables
            outs.append(key_results[0])
        else:
            outs.append(self.concat_one_mode_results(key_results))
        outs.append(self.concat_one_mode_results(reference_results))

        return outs


@TRANSFORMS.register_module()
class ConcatVideoReferences(ConcatSameTypeFrames):
    """Concat video references.

    If the input list contains at least two dicts, concat the input list of
    dict to one dict from 2-nd dict of the input list.

    Note: the 'ConcatVideoReferences' class will be deprecated in the
    future, please use 'ConcatSameTypeFrames' instead.
    """

    def __init__(self):
        warnings.warn(
            "The 'ConcatVideoReferences' class will be deprecated in the "
            "future, please use 'ConcatSameTypeFrames' instead")
        super(ConcatVideoReferences, self).__init__(num_key_frames=1)


@TRANSFORMS.register_module()
class MultiImagesToTensor(BaseTransform):
    """Multi images to tensor.

    1. Transpose and convert image/multi-images to Tensor.
    2. Add prefix to every key in the second dict of the inputs. Then, add
    these keys and corresponding values into the outputs.

    Args:
        ref_prefix (str): The prefix of key added to the second dict of inputs.
            Defaults to 'ref'.
    """

    def __init__(self, ref_prefix='ref'):
        self.ref_prefix = ref_prefix

    def __call__(self, results):
        """Multi images to tensor.

        1. Transpose and convert image/multi-images to Tensor.
        2. Add prefix to every key in the second dict of the inputs. Then, add
        these keys and corresponding values into the output dict.

        Args:
            results (list[dict]): List of two dicts.

        Returns:
            dict: Each key in the first dict of `results` remains unchanged.
            Each key in the second dict of `results` adds `self.ref_prefix`
            as prefix.
        """
        outs = []
        for _results in results:
            _results = self.images_to_tensor(_results)
            outs.append(_results)

        data = {}
        data.update(outs[0])
        if len(outs) == 2:
            for k, v in outs[1].items():
                data[f'{self.ref_prefix}_{k}'] = v

        return data

    def images_to_tensor(self, results):
        """Transpose and convert images/multi-images to Tensor."""
        if 'img' in results:
            img = results['img']
            if len(img.shape) == 3:
                # (H, W, 3) to (3, H, W)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                # (H, W, 3, N) to (N, 3, H, W)
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results['img'] = to_tensor(img)
        if 'proposals' in results:
            results['proposals'] = to_tensor(results['proposals'])
        if 'img_metas' in results:
            results['img_metas'] = to_tensor(results['img_metas'])
        return results

@TRANSFORMS.register_module()
class CheckPadMaskValidity(object):
    """Check the validity of data. Generally, it's used in such case: The image
    padding masks generated in the image preprocess need to be downsampled, and
    then passed into Transformer model, like DETR. The computation in the
    subsequent Transformer model must make sure that the values of downsampled
    mask are not all zeros.

    Args:
        stride (int): the max stride of feature map.
    """

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): Result dict contains the data to be checked.

        Returns:
            dict | None: If invalid, return None; otherwise, return original
                input.
        """
        for _results in results:
            assert 'padding_mask' in _results
            mask = _results['padding_mask'].copy().astype(np.float32)
            img_h, img_w = _results['img'].shape[:2]
            feat_h, feat_w = img_h // self.stride, img_w // self.stride
            downsample_mask = cv2.resize(
                mask, dsize=(feat_h, feat_w)).astype(bool)
            if (downsample_mask == 1).all():
                return None
        return results


@TRANSFORMS.register_module()
class ToList(object):
    """Use list to warp each value of the input dict.

    Args:
        results (dict): Result dict contains the data to convert.

    Returns:
        dict: Updated result dict contains the data to convert.
    """

    def __call__(self, results):
        out = {}
        for k, v in results.items():
            out[k] = [v]
        return out


@TRANSFORMS.register_module()
class MultiPackSegInputs(PackSegInputs):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        outs = []
        for _results in results:
            _results = super().transform(_results)
            outs.append(_results)
        return outs

