#cuda driver version and cuda toolkit version must match

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install mmcls==0.25.0 openmim scipy scikit-learn ftfy regex tqdm
mim install mmpretrain
mim install mmcv-full==1.6.0
pip install yapf==0.40.1

# pip install openmim
# git clone https://github.com/open-mmlab/mmpretrain.git
# cd mmpretrain
# pip install -U openmim && mim install -e .