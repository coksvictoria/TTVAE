import os
import time
from numpy.lib.type_check import real_if_close

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
import numpy as np

import argparse
import warnings
import json
import pickle

from baselines.ctabgan.model import CTABGANSynthesizer
from baselines.ctabgan.util import DataPrep

warnings.filterwarnings('ignore')

INFO_PATH = 'data_profile'

class CTABGAN():

    def __init__(self,
                 df,
                 test_ratio = 0.20,
                 categorical_columns = [],
                 log_columns = [],
                 mixed_columns= {},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type= {},
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=150,
                 lr=2e-4
                 ):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(
                class_dim=class_dim,
                random_dim=random_dim,
                num_channels=num_channels,
                l2scale=l2scale,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs,
        )
        self.raw_df = df
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
                
    def fit(self):
        
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, num_samples):
        
        sample = self.synthesizer.sample(num_samples) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df

def train(args): 
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'
    num_epochs= args.sdv_epochs
    
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

    with open(f"{curr_dir}/configs/features.json", "r") as f:
        configs = json.load(f)
        
    ctabgan_params=configs[dataname]


    synthesizer =  CTABGAN(
                        df = real,
                        test_ratio = 0.0,  
                        **ctabgan_params,
                        epochs=num_epochs
                    ) 
        
    synthesizer.fit()

    with open(ckpt_path + "/ctabgan.pkl", "wb") as f:
        pickle.dump(synthesizer, f)

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
    with open(ckpt_path + "/ctabgan.pkl",'rb')  as f:
        synthesizer = pickle.load(f)

    syn_df=synthesizer.generate_samples(n_samples)
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