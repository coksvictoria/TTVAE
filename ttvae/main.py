import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np

import argparse
import warnings
import json

from tqdm import tqdm
from ttvae.model import TTVAE


warnings.filterwarnings('ignore')

INFO_PATH = 'data_profile'

def train(args): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'
    num_epochs= args.ttvae_epochs
    
    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)

    column_names=info['column_names']
    print(column_names)

    c_col_idx=info['cat_col_idx']
    c_col=list(np.array(column_names)[c_col_idx])

    target_col_idx=info['target_col_idx']

    if info['task_type']!="regression":
      c_col_idx=c_col_idx+target_col_idx
      c_col=list(np.array(column_names)[c_col_idx])
    
    print(c_col)

    ckpt_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'
    real = pd.read_csv(real_data_path+'/train.csv')

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    model=TTVAE(verbose=True,epochs=num_epochs)
    model.fit(real,c_col,ckpt_path)

def sample(args): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    save_path = args.save_path
    ckpt_path = f'{curr_dir}/ckpt/{dataname}'

    dataset_dir = f'data/{dataname}'
    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    n_samples=info['train_num']
    start_time = time.time()

    model=torch.load(ckpt_path+'/model.pt')
    syn_df=model.sample(n_samples)

    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TTVAE')
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'