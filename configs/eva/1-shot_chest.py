_base_ = [
    '../_base_/datasets/chest.py',
    '../_base_/schedules/chest.py',
    '../_base_/default_runtime.py', 
    '../_base_/custom_imports.py'
]

lr = 1e-3
vpl = 1
dataset = 'chest'
exp_num = 2
nshot = 1

run_name = f'eva02-b_{vpl}_bs4_lr{lr}_{nshot}-shot_{dataset}_exp{exp_num}'
work_dir = f'work_dirs/chest/{nshot}-shot/{run_name}'

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PromptedViTEVA02',
        prompt_length=vpl,
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
            prefix='backbone',),
        ),
    neck=None,
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=19,
        in_channels=768,
    ))