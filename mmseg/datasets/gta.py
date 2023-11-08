# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from .cityscapes import CityscapesDataset
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class GTADataset(BaseSegDataset):

    METAINFO = CityscapesDataset.METAINFO

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)