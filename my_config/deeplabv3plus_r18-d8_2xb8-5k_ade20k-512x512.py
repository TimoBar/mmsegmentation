_base_ = [
    '../configs/_base_/models/deeplabv3plus_r50-d8.py',
    '../configs/_base_/datasets/ade20k.py', '../configs/_base_/default_runtime.py',
    '../configs/_base_/schedules/tiny_schedule.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             decode_head=dict(num_classes=150),
             auxiliary_head=dict(num_classes=150))
default_hooks = dict(logger = dict(type='LoggerHook', interval=1, log_metric_by_epoch=False))
val_dataloader = dict(dataset=dict(indices=3))  # limit val data
train_dataloader = dict(batch_size=2)