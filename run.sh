export WORKDIR=/medfmc_exp
cd $WORKDIR
export PYTHONPATH=$PWD:$PYTHONPATH

#20 epoch
python tools/train.py configs/convnext/5-shot_colon.py

python tools/train.py configs/resnet/5-shot_colon.py

python tools/train.py configs/densenet121/5-shot_colon.py

python tools/train.py configs/efficientnetv2_large/5-shot_colon.py

python tools/train.py configs/swin_v2_base/5-shot_colon.py

python tools/train.py configs/swin_v2_large/5-shot_colon.py

#50 epoch

#100 epoch


# python tools/train.py configs/resnet/5-shot_colon.py
# Training

# Evaluation
# python tools/test.py configs/swin_v2_small/10-shot_chest.py /work_dirs/chest/10-shot/swinv2-t_bs2_lr1e-06_exp1/10-shot_chest.py
