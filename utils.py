from baselines.shallow.main import train_smote
from baselines.shallow.main import train_synthpop
from baselines.shallow.main import train_copula

from baselines.sdv.main import train_ctgan
from baselines.sdv.main import train_tvae
from baselines.sdv.main import train_copulagan

from baselines.ctabgan.main import train as train_ctabgan
from baselines.ctabgan.main import sample as sample_ctabgan

from baselines.stasy.main import main as train_stasy
from baselines.tabddpm.main_train import main as train_tabddpm

from baselines.stasy.sample import main as sample_stasy
from baselines.tabddpm.main_sample import main as sample_tabddpm

from baselines.tabsyn.vae.main import main as train_vae
from baselines.tabsyn.main import main as train_tabsyn
from baselines.tabsyn.sample import main as sample_tabsyn

from ttvae.main import train as train_ttvae
from ttvae.main import sample as sample_ttvae


import argparse
import importlib
import ml_collections

def execute_function(method, mode):
    if method == 'vae':
        mode = 'train'

    main_fn = eval(f'{mode}_{method}')

    return main_fn

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='tabsyn', help='Method: tabsyn or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')

    ''' configs for SDV + CTABGAN '''

    parser.add_argument('-e', '--sdv_epochs', default=300, type=int, help='Number of training epochs for CTGAN,TVAE,CopulaGAN, CTABGAN')
    # parser.add_argument('--no-header', dest='header', action='store_false',
    #                     help='The CSV file has no header. Discrete columns will be indices.')

    # parser.add_argument('-m', '--metadata', help='Path to the metadata')
    # parser.add_argument('-d', '--discrete',
    #                     help='Comma separated list of discrete columns without whitespaces.')
    # parser.add_argument('-n', '--num-samples', type=int,
    #                     help='Number of rows to sample. Defaults to the training data size')

    # parser.add_argument('--generator_lr', type=float, default=2e-4,
    #                     help='Learning rate for the generator.')
    # parser.add_argument('--discriminator_lr', type=float, default=2e-4,
    #                     help='Learning rate for the discriminator.')

    # parser.add_argument('--generator_decay', type=float, default=1e-6,
    #                     help='Weight decay for the generator.')
    # parser.add_argument('--discriminator_decay', type=float, default=0,
    #                     help='Weight decay for the discriminator.')

    # parser.add_argument('--embedding_dim', type=int, default=1024,
    #                     help='Dimension of input z to the generator.')
    # parser.add_argument('--generator_dim', type=str, default='1024,2048,2048,1024',
    #                     help='Dimension of each generator layer. '
    #                     'Comma separated integers with no whitespaces.')
    # parser.add_argument('--discriminator_dim', type=str, default='1024,2048,2048,1024',
    #                     help='Dimension of each discriminator layer. '
    #                     'Comma separated integers with no whitespaces.')

    # parser.add_argument('--save', default=None, type=str,
    #                     help='A filename to save the trained synthesizer.')
    # parser.add_argument('--load', default=None, type=str,
    #                     help='A filename to load a trained synthesizer.')

    # parser.add_argument('--sample_condition_column', default=None, type=str,
    #                     help='Select a discrete column name.')
    # parser.add_argument('--sample_condition_column_value', default=None, type=str,
    #                     help='Specify the value of the selected discrete column.')


    # configs for traing StaSy
    parser.add_argument('--stasy_epochs', type=int, default=100, help='Number of training epochs for StaSy')

    # configs for traing TabSyn's VAE
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Maximum beta')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Batch size.')
    parser.add_argument('--vae_epochs', type=int, default=100, help='Number of training epochs for VAE')

    parser.add_argument('--tabsyn_epochs', type=int, default=100, help='Number of training epochs for Tabsyn')

    # configs for TabDDPM
    parser.add_argument('--ddim', action = 'store_true', default=False, help='Whether use DDIM sampler')
    parser.add_argument('--tabddpm_epochs', type=int, default=10000, help='Number of training epochs for TabDDPM')

    # configs for TTVAE
    parser.add_argument('--ttvae_epochs', type=int, default=100, help='Number of training epochs for TTVAE')
    parser.add_argument('--lsi_method', type=str, default='triangle', help='Latent space interpolation method')

    # configs for sampling in general
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')
    parser.add_argument('--steps', type=int, default=50, help='NFEs.')


    
    args = parser.parse_args()

    return args