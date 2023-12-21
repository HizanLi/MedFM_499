# export WORKDIR=/medfmc_exp
# cd $WORKDIR
# export PYTHONPATH=$PWD:$PYTHONPATH
# # Training
# python tools/train.py configs/swin_v2_small/10-shot_chest.py

Evaluation
python tools/test.py configs/swin_v2_small/10-shot_chest.py /work_dirs/chest/10-shot/swinv2-t_bs2_lr1e-06_exp1/10-shot_chest.py
