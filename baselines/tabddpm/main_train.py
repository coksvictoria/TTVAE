import os
import argparse

from baselines.tabddpm.train import train

import src


def main(args):

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataname = args.dataname
    device = f'cuda:{args.gpu}'

    config_path = f'{curr_dir}/configs/{dataname}.toml'
    model_save_path = f'{curr_dir}/ckpt/{dataname}'
    real_data_path = f'data/{dataname}'

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    args.train = True
    raw_config = src.load_config(config_path)

    ''' 
    Modification of configs
    '''
    print('START TRAINING')

    train(
        model_save_path=model_save_path,
        real_data_path=real_data_path,
        task_type=raw_config['task_type'],
        model_type=raw_config['model_type'],
        model_params=raw_config['model_params'],
        T_dict=raw_config['train']['T'],
        steps=args.tabddpm_epochs,
        device=device
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE')
    parser.add_argument('--dataname', type = str, default = 'adult')
    parser.add_argument('--gpu', type = int, default=0)
    parser.add_argument('--tabddpm_epochs', type = int, default=10000)

    args = parser.parse_args()