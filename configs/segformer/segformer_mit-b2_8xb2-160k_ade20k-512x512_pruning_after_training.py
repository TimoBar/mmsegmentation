_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512_pruning_after_training.py']

checkpoint = './pretrained_weight/segformer/mit_b2_prunable_20220624-66e8bf70.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
