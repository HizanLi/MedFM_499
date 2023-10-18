dataset_type = 'Chest19'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=384, backend='pillow', interpolation='bicubic'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='Chest19',
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data/MedFMC_train/chest/train_20.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=384,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='Chest19',
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data/MedFMC_train/chest/val_20.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=384,
                backend='pillow',
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='Chest19',
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data/MedFMC_train/chest/test_WithLabel.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                size=384,
                backend='pillow',
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='mAP', save_best='auto')
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.005,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=1,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=1)
]
train_cfg = dict(by_epoch=True, val_interval=1, max_epochs=200)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=1024)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_imports = dict(
    imports=[
        'medfmc.models', 'medfmc.datasets.medical_datasets',
        'medfmc.core.evaluation'
    ],
    allow_failed_imports=False)
lr = 0.001
vpl = 1
dataset = 'chest'
exp_num = 2
nshot = 1
run_name = 'eva02-b_1_bs4_lr0.001_1-shot_chest_exp2'
work_dir = 'work_dirs/chest/1-shot/eva02-b_1_bs4_lr0.001_1-shot_chest_exp2'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedViTEVA02',
        prompt_length=1,
        patch_size=14,
        sub_ln=True,
        final_norm=False,
        out_type='avg_featmap',
        arch='b',
        img_size=448,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpretrain/v1.0/eva02/eva02-base-p14_in21k-pre_in21k-medft_3rdparty_in1k-448px_20230505-5cd4d87f.pth',
            prefix='backbone')),
    neck=None,
    head=dict(type='MultiLabelLinearClsHead', num_classes=19, in_channels=768))
gpu_ids = [0]
