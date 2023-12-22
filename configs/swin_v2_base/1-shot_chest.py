_base_ = [
    '../_base_/datasets/chest.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    'mmpretrain::_base_/default_runtime.py',
    '../_base_/custom_imports.py',
]

lr = 1e-6
train_bs = 2
val_bs = 32
dataset = 'chest'
model_name = 'swin_v2_base'
exp_num = 1
nshot = 1
seed = 2049
randomness = dict(seed=seed)

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    _scope_='mmpretrain',
    backbone=dict(
        arch='base',
        drop_path_rate=0.2,
        img_size=384,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w24_in21k-pre_3rdparty_in1k-384px_20220803-44eb70f8.pth',
            prefix='backbone',
            type='Pretrained'),
        pretrained_window_sizes=[12, 12, 12, 6],
        type='SwinTransformerV2',
        window_size=[24, 24, 24, 12]),
    neck=None,
    head=dict(
        type='CSRAClsHead',
        num_classes=19,
        in_channels=1024,
        num_heads=1,
        lam=0.1,
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    type='ImageClassifier')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(type='torchvision/RandomAffine', degrees=(-15, 15), translate=(0.05, 0.05), fill=128),
    dict(type='PILToNumpy', to_bgr=True),
    dict(type='RandomResizedCrop', scale=384, crop_ratio_range=(0.9, 1.0), backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=train_bs,
    dataset=dict(ann_file=f'data/MedFMC_train/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'),
)

val_dataloader = dict(
    batch_size=val_bs,
    dataset=dict(ann_file=f'data/MedFMC_train/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
)

test_dataloader = dict(
    batch_size=8,
    dataset=dict(ann_file=f'data/MedFMC_train/{dataset}/test_WithLabel.txt'),
)

optimizer = dict(betas=(0.9, 0.999), eps=1e-08, lr=lr, type='AdamW', weight_decay=0.05)

optim_wrapper = dict(
    optimizer=optimizer,
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

param_scheduler = [
    dict(by_epoch=True, end=1, start_factor=1, type='LinearLR'),
    dict(begin=1, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

train_cfg = dict(by_epoch=True, val_interval=25, max_epochs=20)

randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=250, max_keep_ckpts=1, save_best="Aggregate", rule="greater"),
    logger=dict(interval=10),
)