_base_ = [
    'mmpretrain::_base_/models/convnext_v2/base.py',
    '../_base_/datasets/colon.py',
    '../_base_/schedules/adamw_inverted_cosine_lr.py',
    'mmpretrain::_base_/default_runtime.py',
    '../_base_/custom_imports.py',
]

lr = 2.5e-3
train_bs = 6
val_bs = 96
dataset = 'colon'
model_name = 'convnext-v2-b'
exp_num = 1
nshot = 5

run_name = f'{model_name}_bs{train_bs}_lr{lr}_exp{exp_num}_'
work_dir = f'work_dirs/{dataset}/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ConvNeXt',
        arch='base',
        init_cfg=dict(
            prefix='backbone',
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
        )
    ),
    head=dict(
        in_channels=1024,
        num_classes=2,
        type='LinearClsHead'
    )
)

# dataset setting
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

visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

# train, val, test setting
train_cfg = dict(by_epoch=True, val_interval=5, max_epochs=20)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=1e-4, priority='ABOVE_NORMAL')]

randomness = dict(seed=0)

default_hooks = dict(
    checkpoint=dict(interval=250, max_keep_ckpts=1, save_best="accuracy/top1", rule="greater"),
    logger=dict(interval=10),
)
