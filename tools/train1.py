# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import re
from copy import deepcopy
from datetime import datetime

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


"""
    Adapted from:
    https://github.com/pytorch/examples/blob/main/vae/main.py
    https://github.com/open-mmlab/mmdetection/blob/main/tools/train.py

"""


def parse_args():
    """
    
    The parse_args() function is designed to parse command-line arguments provided when running the script. 
    Its purpose is to allow users to specify various configuration options when training a model. 
    
    """
        
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    #-----------------------------------------------------------------------------------------#
    parser.add_argument('--remove_timestamp',
                        action='store_true', help='Remove timestamp from work_dir')
    parser.add_argument('--exp_suffix', type=str,
                        default='', help='Suffix for experiment name')
    parser.add_argument('--lr', default=None, type=float,
                        help='Override the learning rate from the config file.')
    parser.add_argument('--exp_num', type=int, default=None,
                        help='Experiment number for data_anns')
    parser.add_argument('--train_bs', type=int, default=None,
                        help='Training batch size.')
    parser.add_argument('--seed', type=int, default=None,
                        help='The seed for training.')
    #-----------------------------------------------------------------------------------------#

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args