_base_ = [
    '../_base_/models/densenet/densenet121_multilabel.py',
    '../_base_/datasets/chest.py', '../_base_/schedules/imagenet_dense.py',
    '../_base_/default_runtime.py', '../_base_/custom_imports.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='pretrain/densenet121_4xb256_in1k_20220426-07450f99.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=19))

dataset = 'chest'
nshot = 5
exp_num = 1
data = dict(
    samples_per_gpu=4,  # use 2 gpus, total 128
    train=dict(
        ann_file=f'data/MedFMC_train/{dataset}/{dataset}_{nshot}-shot_train_exp{exp_num}.txt'),
    val=dict(ann_file=f'data/MedFMC_train/{dataset}/{dataset}_{nshot}-shot_val_exp{exp_num}.txt'),
    test=dict(ann_file=f'data/MedFMC_train/{dataset}/test_WithLabel.txt'))
