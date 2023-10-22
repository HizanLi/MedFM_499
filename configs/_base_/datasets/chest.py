# dataset settings
dataset_type = 'Chest19'
data_preprocessor = dict(
    num_classes=19,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True
)

"""
    https://mmdetection.readthedocs.io/en/v2.19.1/tutorials/data_pipeline.html
    https://mmpretrain.readthedocs.io/en/stable/api/data_process.html
"""
#---------------------------------------------------------------#
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='NumpyToPIL', to_rgb=True),
    dict(type='torchvision/RandomAffine', degrees=(-15, 15), translate=(0.05, 0.05), fill=128),
    dict(type='PILToNumpy', to_bgr=True),
    dict(type='RandomResizedCrop', scale=384, crop_ratio_range=(0.9, 1.0), backend='pillow', interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
#---------------------------------------------------------------#

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=384, backend='pillow', interpolation='bicubic'),
    dict(type='Normalize', **data_preprocessor),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data/MedFMC_train/chest/train_20.txt',
        pipeline=train_pipeline,),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data/MedFMC_train/chest/val_20.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/MedFMC_train/chest/images',
        ann_file='data/MedFMC_train/chest/test_WithLabel.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

train_evaluator = [
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),  # class-wise mean
    dict(type='MultiLabelMetric', average='micro'),  # overall mean
    dict(type='AUC', multilabel=True),
    dict(type='Aggregate', multilabel=True)]
val_evaluator = train_evaluator
test_evaluator = train_evaluator
