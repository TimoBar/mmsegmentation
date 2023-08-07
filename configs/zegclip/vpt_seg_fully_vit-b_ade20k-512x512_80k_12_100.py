_base_ = [
    '../_base_/models/zegclip.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

img_size = 512
in_channels = 512
out_indices = [11]
crop_size = (img_size, img_size)
find_unused_parameters = True
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

base_class = [0, 1, 2, 3, 4, 5] #, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
novel_class = []
both_class = base_class
num_classes = len(base_class)

"""train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(img_size, img_size), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]"""

pretrained = './pretrained_weight/zegclip/ViT-B-16.pt'

model = dict(
    type='ZegCLIP',
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    pretrained_text=pretrained,
    context_length=77,
    backbone=dict(
        type='VPTCLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=img_size,
        out_indices=out_indices,
        #setting of vpt
        num_tokens=100,
        prompt_dim=768,
        total_d_layer=11,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextEncoder',
        context_length=77,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    decode_head=dict(
        type='ATMSingleHeadSeg',
        img_size=img_size,
        in_channels=in_channels,
        seen_idx=base_class,
        all_idx=both_class,
        channels=in_channels,
        num_classes=num_classes,
        num_layers=3,
        num_heads=8,
        use_proj=False,
        use_stages=len(out_indices),
        embed_dims=in_channels,
        loss_decode=dict(
            type='SegLossPlus', num_classes=num_classes, dec_layers=3,
            mask_weight=2.0,
            dice_weight=1.0,
            loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(img_size, img_size), stride=(426, 426)),
    base_class = base_class,
    novel_class = novel_class,
    both_class = both_class,
    ft_backbone = False,
    #class_names = ["wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chestofdrawers", "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pooltable", "pillow", "screendoor", "stairway", "river", "bridge", "bookcase", "blind", "coffeetable", "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchenisland", "computer", "swivelchair", "boat", "bar", "arcademachine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "televisionreceiver", "airplane", "dirttrack", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyerbelt", "canopy", "washer", "plaything", "swimmingpool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step", "tank", "tradename", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase", "trafficlight", "tray", "ashcan", "fan", "pier", "crtscreen", "plate", "monitor", "bulletinboard", "shower", "radiator", "glass", "clock", "flag"],
    class_names = ["dog", "cat", "car", "floor", "sky", "human"],
    exclude_key='prompt',
    load_text_embedding=None
    #load_text_embedding='configs/_base_/datasets/text_embedding/voc12_single.npy'
)
#class names voc
"""class_names = ['aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],"""

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00002, weight_decay=0.01,
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=10.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.),
                                        'ln': dict(decay_mult=0.),
                                        'head': dict(lr_mult=10.),
                                        }))

train_dataloader = dict(batch_size=2, num_workers=2)
#data = dict(samples_per_gpu=4,
#            workers_per_gpu=4,)
