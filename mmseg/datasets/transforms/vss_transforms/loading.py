# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import LoadAnnotations, LoadImageFromFile
import os.path

@TRANSFORMS.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    """Load multi images from file.

    Please refer to `mmdet.datasets.TRANSFORMS.loading.py:LoadImageFromFile`
    for detailed docstring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in `results`, call the call function of
        `LoadImageFromFile` to load image.

        Args:
            results (list[dict]): List of dict from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded image.
        """
        outs = []
        for _results in results:
            _results = super().__call__(_results)
            outs.append(_results)
        return outs




@TRANSFORMS.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    """Sequence load annotations.

    Please refer to `mmdet.datasets.TRANSFORMS.loading.py:LoadAnnotations`
    for detailed docstring.

    Args:
        with_track (bool): If True, load instance ids of bboxes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function.

        For each dict in results, call the call function of `LoadAnnotations`
        to load annotation.

        Args:
            results (list[dict]): List of dict that from
                :obj:`mmtrack.CocoVideoDataset`.

        Returns:
            list[dict]: List of dict that contains loaded annotations, such as
            bounding boxes, labels, instance ids, masks and semantic
            segmentation annotations.
        """
        outs = []
        for _results in results:
            if os.path.isfile(_results['seg_map_path']):
                _results = super().__call__(_results)
            outs.append(_results)
        return outs
