# you need to export path in terminal so the `custom_imports` in config would work
export PYTHONPATH=$PWD:$PYTHONPATH
# Training
# you can choose a config file like `configs/densenet/dense121_chest.py to train its model
python tools/train.py configs/densenet/dense121_chest.py

# Evaluation
# Endo and ChestDR utilize mAP as metric
python tools/test.py configs/densenet/dense121_chest.py work_dirs/dense121_chest/latest.pth --metrics mAP
python tools/test.py configs/densenet/dense121_chest.py work_dirs/dense121_chest/latest.pth --metrics AUC_multilabel

# # Colon utilizes accuracy as metric
# python tools/test.py configs/densenet/dense121_chest.py work_dirs/dense121_chest/latest.pth --metrics accuracy --metric-options topk=1
# python tools/test.py configs/densenet/dense121_chest.py work_dirs/dense121_chest/latest.pth --metrics AUC_multiclass
