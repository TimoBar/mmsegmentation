_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(size=crop_size)
checkpoint = './pretrained_weight/segformer/mit_b0_prunable_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    type="PrunedEncoderDecoder",
    mask_factor=0.1,
    data_preprocessor=data_preprocessor,
    backbone=dict(type='MixVisionTransformerPrunable', init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    decode_head=dict(type="SegformerHeadPrunable"),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.000015, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
            'p1': dict(lr_mult=10.)#, weight_decay=0),
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500)

]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
