import os

from imblearn.under_sampling import TomekLinks,EditedNearestNeighbours,RandomUnderSampler
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SVMSMOTE,SMOTE,KMeansSMOTE,SMOTENC
from imblearn.combine import SMOTEENN,SMOTETomek

from imblearn.pipeline import Pipeline as imbPipeline

from synthpop import CAT_COLS_DTYPES, Synthpop

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

import torch
import argparse
import warnings
import time
import json
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
INFO_PATH = 'data_profile'

def encode_df(df_raw,c_col):
    df=df_raw.copy()
    cat_dict={}
    for i in c_col:
      df[i]=df[i].astype('category')
      cat_dict[i] = dict(enumerate(df[i].cat.categories))
      df[i]=df[i].cat.codes
      df[i]=df[i].astype('int')

    return df,cat_dict

def check_integer(x):
    try:
        pd.to_numeric(x)
    except (RuntimeError, TypeError, NameError, IOError, ValueError):
        return False
    else:
        if (x.dropna() % 1 == 0).all():
          return True
        else:
          return False
          


def train_smote(args): 
    device = args.device
    num_epochs = args.sdv_epochs + 1
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    real_data_path = f'data/{dataname}'
    real_raw = pd.read_csv(real_data_path+'/train.csv')

    n_samples=real_raw.shape[0]

    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)

    column_names=info['column_names']

    c_col_idx=info['cat_col_idx']
    c_col=list(np.array(column_names)[c_col_idx])

    target_col_idx=info['target_col_idx']

    if info['task_type']!="regression":
      c_col_idx=c_col_idx+target_col_idx
      c_col=list(np.array(column_names)[c_col_idx])
    
    n_col= list(set(column_names) - set(c_col))

    real,cat_dict=encode_df(real_raw,c_col)

    int_s=real_raw.apply(check_integer)
    int_col=int_s[int_s].index.to_list()

    df_real=real.copy()
    df_fake=real.copy()
    df_real['real']=1
    df_fake['real']=0
    df_rff=pd.concat([df_real,df_fake,df_fake])
    y_f=df_rff['real']
    X_f=df_rff.drop(['real'],axis=1)

    ratio=1

    balancer_names=[
        "SMOTE",
        # "SMOTENC",
        "ADASYN",
        "SMOTETomek"
        ]
    balancers = [
                SMOTE(sampling_strategy=ratio,random_state=1),
                # SMOTENC(sampling_strategy=ratio,random_state=1, categorical_features=c_col_idx),
                ADASYN(sampling_strategy=ratio,random_state=1),
                SMOTETomek(smote=SMOTE(sampling_strategy=ratio,random_state=1)),
                ]

    zipped_balancer = zip(balancer_names,balancers)
    for n,o in zipped_balancer:
      print('-----------'+n+'----------------')
      pipe = imbPipeline([('over', o)])
      start_time = time.time()
      X_o,y_o=pipe.fit_resample(X_f,y_f)
      y_o=y_o.astype('int')
      df_o=np.c_[X_o, y_o]
      df_new=pd.DataFrame(df_o)
      synthetic=df_new.iloc[-n_samples:,:-1]
      synthetic.columns=real.columns
      print(synthetic.shape)
      end = time.time()

      for col in c_col:
        synthetic[col] = synthetic[col].astype('object')
        synthetic[col] = synthetic[col].map(cat_dict[col])

      for i in int_col:
        synthetic[i]=synthetic[i].astype('int')

      save_path= f'synthetic/{args.dataname}/{n}.csv'
      synthetic.to_csv(save_path, index = False)

      end_time = time.time()
      print('Time: ', end_time - start_time)

      print('Saving sampled data to {}'.format(save_path))

def train_synthpop(args): 
    device = args.device
    num_epochs = args.sdv_epochs + 1
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    real_data_path = f'data/{dataname}'
    real_raw = pd.read_csv(real_data_path+'/train.csv')

    n_samples=real_raw.shape[0]

    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)

    column_names=info['column_names']

    c_col_idx=info['cat_col_idx']
    c_col=list(np.array(column_names)[c_col_idx])

    target_col_idx=info['target_col_idx']

    if info['task_type']!="regression":
      c_col_idx=c_col_idx+target_col_idx
      c_col=list(np.array(column_names)[c_col_idx])
    
    n_col= list(set(column_names) - set(c_col))

    real,cat_dict=encode_df(real_raw,c_col)

    int_s=real_raw.apply(check_integer)
    int_col=int_s[int_s].index.to_list()

    start_time = time.time()
    dtypes={}
    for i in real.columns:
      if i in c_col:
        dtypes[i]="category"
      elif i in int_col:
        dtypes[i]="int"
      else:
        dtypes[i]="float"

    spop = Synthpop()
    spop.fit(real,dtypes)

    synthetic = spop.generate(n_samples)

    for col in c_col:
      synthetic[col] = synthetic[col].astype('object')
      synthetic[col] = synthetic[col].map(cat_dict[col])

    for i in int_col:
      synthetic[i]=synthetic[i].astype('int')

    save_path= f'synthetic/{args.dataname}/{args.method}.csv'
    synthetic.to_csv(save_path, index = False)

    end_time = time.time()
    print('Time: ', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))


def train_copula(args): 
    device = args.device
    num_epochs = args.sdv_epochs + 1
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    real_data_path = f'data/{dataname}'
    real_raw = pd.read_csv(real_data_path+'/train.csv')

    n_samples=real_raw.shape[0]

    with open(f'{INFO_PATH}/{dataname}.json', 'r') as f:
        info = json.load(f)

    column_names=info['column_names']

    c_col_idx=info['cat_col_idx']
    c_col=list(np.array(column_names)[c_col_idx])

    target_col_idx=info['target_col_idx']

    if info['task_type']!="regression":
      c_col_idx=c_col_idx+target_col_idx
      c_col=list(np.array(column_names)[c_col_idx])
    
    n_col= list(set(column_names) - set(c_col))

    real,cat_dict=encode_df(real_raw,c_col)

    int_s=real_raw.apply(check_integer)
    int_col=int_s[int_s].index.to_list()

    start_time = time.time()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_raw)


    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real)

    synthetic = synthesizer.sample(num_rows=n_samples)

    for col in c_col:
      synthetic[col] = synthetic[col].astype('object')
      synthetic[col] = synthetic[col].map(cat_dict[col])

    for i in int_col:
      synthetic[i]=synthetic[i].astype('int')

    save_path= f'synthetic/{args.dataname}/{args.method}.csv'
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