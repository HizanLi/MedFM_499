export WORKDIR=/medfmc_exp
cd $WORKDIR
export PYTHONPATH=$PWD:$PYTHONPATH
# Training
python tools/train.py configs/prompt_vit/1-shot_chest.py

# Evaluation
# python tools/test.py configs/eva-b_vpt/1-shot_endo.py work_dirs/endo/1-shot/eva02-b_1_bs4_lr0.001_1-shot_endo_exp3_20231019-231804/epoch_20.pth
# python tools/test.py configs/eva-b_vpt/10-shot_endo.py work_dirs/endo/10-shot/eva02-b_1_bs4_lr0.001_10-shot_endo_exp1_20231019-191708/best_multi-label_mAP_epoch_16.pth
