import os
from ctgan import CTGAN,TVAE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer

import torch
import argparse
import warnings
import time
import json
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
INFO_PATH = 'data_profile'

def train_ctgan(args): 
    device = args.device
    num_epochs = args.sdv_epochs + 1
    dataname = args.dataname
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    real_data_path = f'data/{dataname}'
    real = pd.read_csv(real_data_path+'/train.csv')

    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)

    column_names=info['column_names']
    # print(column_names)

    c_col_idx=info['cat_col_idx']
    c_col=list(np.array(column_names)[c_col_idx])

    target_col_idx=info['target_col_idx']

    if info['task_type']!="regression":
      c_col_idx=c_col_idx+target_col_idx
      c_col=list(np.array(column_names)[c_col_idx])
    
    # print(c_col)

    ckpt_path = f'{curr_dir}/ctgan/{dataname}'
    # print(ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    start_time = time.time()
    ctgan = CTGAN(verbose=True,epochs=num_epochs)

    ctgan.fit(real, c_col)
    ctgan.save(ckpt_path+'/CTGAN.pkl')
  
    ctgan_load = CTGAN.load(ckpt_path+'/CTGAN.pkl')
    synthetic = ctgan_load.sample(real.shape[0])

    synthetic.to_csv(save_path, index = False)

    end_time = time.time()
    print('Time: ', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

def train_tvae(args): 
    device = args.device
    num_epochs = args.sdv_epochs + 1
    dataname = args.dataname
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    real_data_path = f'data/{dataname}'
    real = pd.read_csv(real_data_path+'/train.csv')


    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)

    column_names=info['column_names']
    # print(column_names)

    c_col_idx=info['cat_col_idx']
    c_col=list(np.array(column_names)[c_col_idx])

    target_col_idx=info['target_col_idx']

    if info['task_type']!="regression":
      c_col_idx=c_col_idx+target_col_idx
      c_col=list(np.array(column_names)[c_col_idx])
    
    # print(c_col)
    
    ckpt_path = f'{curr_dir}/tvae/{dataname}'
    # print(ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    start_time = time.time()
    tvae = TVAE(epochs=num_epochs)

    tvae.fit(real, c_col)
    tvae.save(ckpt_path+'/TVAE.pkl')
  
    tvae_load = TVAE.load(ckpt_path+'/TVAE.pkl')
    synthetic = tvae_load.sample(real.shape[0])

    synthetic.to_csv(save_path, index = False)

    end_time = time.time()
    print('Time: ', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))


def train_copulagan(args): 
    device = args.device
    num_epochs = args.sdv_epochs + 1
    dataname = args.dataname
    save_path = args.save_path

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    real_data_path = f'data/{dataname}'
    real = pd.read_csv(real_data_path+'/train.csv')

    ckpt_path = f'{curr_dir}/copulagan/{dataname}'
    # print(ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    start_time = time.time()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real)

    copulagan = CopulaGANSynthesizer(metadata,epochs=num_epochs)

    copulagan.fit(real)
    copulagan.save(ckpt_path+'/CopulaGAN.pkl')
  
    copulagan_load = CopulaGANSynthesizer.load(ckpt_path+'/CopulaGAN.pkl')
    synthetic = copulagan_load.sample(real.shape[0])

    synthetic.to_csv(save_path, index = False)

    end_time = time.time()
    print('Time: ', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'