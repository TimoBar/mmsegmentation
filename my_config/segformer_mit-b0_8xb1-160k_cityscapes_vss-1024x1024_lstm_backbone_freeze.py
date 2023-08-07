_base_ = [
    '../configs/_base_/models/segformer_mit-b0.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_160k.py'
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=5000)
find_unused_parameters = True
#SegFormer MiTB5 as teacher
norm_cfg_teacher = dict(type='SyncBN', requires_grad=True)


crop_size = (1024, 1024)
data_preprocessor = dict(type="SeqSegDataPreProcessor", size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    type='VSSEncoderDecoder',
    teacher=None,
    use_clstm_neck=True,
    freeze_backbone=True,
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# dataset settings
dataset_type = 'CityscapesVideoDataset'
data_root = '/beegfs/data/shared/cityscapes_sequence'
crop_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations'),
    dict(
        type='SeqRandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='SeqRandomCrop', crop_size=crop_size, share_params=True),
    dict(type='SeqRandomFlip', prob=0.5),
    dict(type='SeqPad', size_divisor=16),
    #dict(type='PhotoMetricDistortion'),
    dict(type='MultiPackSegInputs')
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='SeqLoadAnnotations'),
    dict(type='MultiPackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        key_frame_index=19,
        frames_before=6,
        frames_after=0,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit_sequence/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        key_frame_index=19,
        frames_before=6,
        frames_after=0,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit_sequence/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

