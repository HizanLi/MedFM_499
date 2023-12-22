#cuda driver version and cuda toolkit version must match

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install mmcls==0.25.0 openmim scipy scikit-learn ftfy regex tqdm
mim install mmpretrain

# pip install openmim
# git clone https://github.com/open-mmlab/mmpretrain.git
# cd mmpretrain
# pip install -U openmim && mim install -e .