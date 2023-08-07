_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [1.286, 1.2669, 1.2503, 1.2176, 1.2187, 1.1893, 1.1887, 1.1683,
                1.1633, 1.147, 1.1488, 1.128, 1.1609, 1.1349, 1.1224, 1.1282,
                1.1502, 1.1245, 1.1164, 1.1193, 1.1154, 1.0985, 1.0941, 1.0834,
                1.0965, 1.1037, 1.1092, 1.0691, 1.0469, 1.0912, 1.0594, 1.0695,
                1.044, 1.0545, 1.0627, 1.0455, 1.0254, 1.0221, 1.0321, 1.0283,
                1.0352, 1.0255, 1.0247, 1.0202, 1.0059, 1.024, 1.0218, 1.0168,
                1.0762, 1.011, 1.0001, 1.0058, 1.0057, 0.9996, 0.9826, 1.0204,
                1.0159, 1.0119, 0.9916, 1.0193, 1.0307, 1.0056, 1.0179, 0.9701,
                0.9948, 0.9864, 1.0021, 1.0083, 1.0075, 0.9999, 0.9773, 0.9901,
                0.9832, 0.9727, 0.9833, 0.9928, 0.9771, 0.9862, 0.9639, 0.9739,
                0.9638, 0.9645, 0.943, 0.9822, 0.9684, 0.954, 0.9651, 0.9398,
                0.9658, 0.9452, 0.9566, 0.9427, 0.9694, 0.9265, 0.965, 0.9577,
                0.9368, 0.9372, 0.9577, 0.9249, 0.9404, 0.9526, 0.9466, 0.9271,
                0.9503, 0.9478, 0.9464, 0.9512, 0.9491, 0.9643, 0.9248, 0.9319,
                0.9224, 0.9742, 0.9629, 0.9199, 0.9124, 0.9465, 0.917, 0.9274,
                0.9453, 0.923, 0.911, 0.918, 0.9125, 0.9144, 0.9258, 0.9071,
                0.9429, 0.9124, 0.9226, 0.8933, 0.9247, 0.8974, 0.8868, 0.9112,
                0.8939, 0.9222, 0.8805, 0.8666, 0.9057, 0.9104, 0.9123, 0.9089,
                0.9111, 0.9038, 0.8757, 0.8829, 0.8897, 0.8873
                ]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'  # noqa
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    #mean=[123.675, 116.28, 103.53],
    #std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=150,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(512 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=2048),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='GenerateEdge', edge_width=4),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

train_dataloader = dict(batch_size=8, dataset=dict(pipeline=train_pipeline))

iters = 960000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]
# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

randomness = dict(seed=304)
