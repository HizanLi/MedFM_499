export WORKDIR=/medfmc_exp
cd $WORKDIR
export PYTHONPATH=$PWD:$PYTHONPATH

python tools/train.py configs/swin_v2_base/10-shot_chest.py

python tools/train.py configs/swin_v2_tiny/10-shot_chest.py

python tools/train.py configs/swin_v2_small/10-shot_chest.py

python tools/train.py configs/swin_v2_large/10-shot_chest.py



# python tools/train.py configs/resnet/10-shot_colon.py --batch_size=2

# python tools/train.py configs/resnet/10-shot_colon.py --batch_size=4

# Training

# Evaluation
# python tools/test.py configs/swin_v2_small/10-shot_chest.py /work_dirs/chest/10-shot/swinv2-t_bs2_lr1e-06_exp1/10-shot_chest.py
