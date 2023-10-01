export WORKDIR=/medfmc_exp
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12
cd $WORKDIR
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/train.py configs/densenet/dense121_chest.py --work-dir work_dirs/temp/
