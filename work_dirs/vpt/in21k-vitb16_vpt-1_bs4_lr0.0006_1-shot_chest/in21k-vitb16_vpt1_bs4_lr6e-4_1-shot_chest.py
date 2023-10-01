custom_imports = dict(
    imports=[
        'medfmc.models', 'medfmc.datasets.medical_datasets',
        'medfmc.core.evaluation'
    ],
    allow_failed_imports=False)
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
        data_prefix='data/MedFMC/chest/images',
        ann_file='data/MedFMC/chest/chest_1-shot_train.txt',
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
        data_prefix='data/MedFMC/chest/images',
        ann_file='data/MedFMC/chest/chest_1-shot_val.txt',
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
        data_prefix='data/MedFMC/chest/images',
        ann_file='data/MedFMC/chest/test_WithLabel.txt',
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
optimizer = dict(type='SGD', lr=0.0006, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth'
resume_from = None
workflow = [('train', 1)]
lr = 0.0006
n = 1
vpl = 1
dataset = 'chest'
nshot = 1
run_name = 'in21k-vitb16_vpt-1_bs4_lr0.0006_1-shot_chest'
model = dict(
    type='ImageClassifier',
    backbone=dict(type='PromptedVisionTransformer', prompt_length=1),
    head=dict(type='MultiLabelLinearClsHead', num_classes=19, in_channels=768))
work_dir = 'work_dirs/vpt/in21k-vitb16_vpt-1_bs4_lr0.0006_1-shot_chest'
gpu_ids = [0]
