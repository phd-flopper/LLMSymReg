import juliacall
from pysr import PySRRegressor
from omegaconf import OmegaConf
from datasets.base import get_dataloader
from datasets.nguyen_dataset import NguyenDataset
from datasets.feynman_dataset import FeynmanDataset
from datasets.feynman_dataset import CompressedFeynmanDataset
from datasets.feynman_dataset import get_compressed_points
from datasets.feynman_dataset import helper_xu
from datasets.feynman_dataset import helper_yu
import numpy as np
from tqdm import tqdm
import logging
import time
import os
import sympy
import gc


def run_all():
    config = OmegaConf.load(PATH)
    if config['main']['run_baseline'] == False:
        return

    run_name = config['baseline']['baseline_run_name']
    run_path = os.path.join(config['run']['run_dir'], run_name)
    os.makedirs(run_path, exist_ok=True)
    
    run_nguyen = config['baseline']['run_nguyen']
    run_feynman = config['baseline']['run_feynman']
    run_feynman_units = config['baseline']['run_feynman_units']
    
    sample_sizes = config['baseline']['sample_sizes']
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S')

    run_logger = logging.getLogger(run_name)
    run_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(run_path + "\log.txt")
    run_logger.addHandler(fh)
    fh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S'))
    
    run_logger.info('Initiating baseline PySR runs.')
    a_st = time.time()
    if run_nguyen:
        run_logger.info('Initiating Nguyen Dataset runs.')
        st = time.time()
        for size in tqdm(sample_sizes):
            sst = time.time()
            run_logger.info(f'Initiating Nguyen_{size} run.')
            lib = {
                   1:[-1, 1, [size, 1]],
                   2:[-1, 1, [size, 1]],
                   3:[-1, 1, [size, 1]],
                   4:[-1, 1, [size, 1]],
                   5:[-1, 1, [size, 1]],
                   6:[-1, 1, [size, 1]],
                   7:[0, 2, [size, 1]],
                   8:[0, 4, [size, 1]],
                   9:[0, 1, [size, 2]],
                   10:[0, 1, [size, 2]],
                   11:[0, 1, [size, 2]],
                   12:[0, 1, [size, 2]],
            }
            operator_library = {
                'unary_operators':['sin', 'cos', 'log', 'exp'], 
                'binary_operators': ['+', '-', '*', '/']
                }
            ext = {'binary_operators': ['+', '-', '*', '/', '^']}
            idx = 1
            n_dicta = {'seed':config['seed'], 'lib':lib}
            n_train_loader = get_dataloader(NguyenDataset, n_dicta)
            exp_logger = logging.getLogger(f'{run_name}_Nguyen_{size}')
            exp_logger.setLevel(logging.INFO)
            nguyen_dir = f"{run_path}\DNguyen\{size}"
            os.makedirs(nguyen_dir, exist_ok=True)
            ffh = logging.FileHandler(f"{nguyen_dir}\log.txt")
            exp_logger.addHandler(ffh)
            ffh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S'))
            
            for x, y, eq in tqdm(n_train_loader):
                eq_st = time.time()
                if idx == 12:
                    operator_library |= ext
                npx = np.asarray(x)
                npx = npx.reshape(npx.shape[-1], npx.shape[0])
                npy = np.asarray(y)
                npy = npy.reshape(npy.shape[-1], npy.shape[0])
                eq = eq[0]
                X = npx
                y = npy
                model = PySRRegressor(**operator_library, verbosity=0, output_directory=nguyen_dir,
                                      run_id = f'{size}_{eq}',
                                      random_state=config['seed'], deterministic=True, parallelism="serial")
                model.fit(X, y)
                idx += 1
                eq_et = time.time()
                exp_logger.info(f'Proposed equation for {eq}:\n {model.sympy()} \n in {eq_et - eq_st} seconds.')
            
            del exp_logger
            eet = time.time()
            run_logger.info(f'Finished Nguyen_{size} run in {eet - sst} seconds.')
        et = time.time()
        run_logger.info(f'Finished Nguyen Dataset runs in {et - st} seconds.')
    
    
    if config['baseline']['get_compressed_points'] == True:
        gcp_st = time.time()
        for size in sample_sizes:
            get_compressed_points(**config['baseline']['gcp'], sample_size=size, seed=config['seed'])
        gcp_et = time.time()
        run_logger.info(f'Built compressed points of specified size in {gcp_et - gcp_st} seconds.')
        
    if run_feynman:
        run_logger.info('Initiating Feynman Dataset runs.')
        st = time.time()
        for size in sample_sizes:
            sst = time.time()
            run_logger.info(f'Initiating Feynman_{size} run.')
            operator_library = {
                'unary_operators':['sin', 'cos', 'log', 'exp', 'sqrt'], 
                'binary_operators': ['+', '-', '*', '/', '^']
                }
            cf_dicta = {'dataset_dir':config['baseline']['cf_dataset_dir'] + f'/Feynman_{size}',
                        'csv_path':config['baseline']['csv_path'],
                        'seed':config['seed']
                        }
            cf_train_loader = get_dataloader(CompressedFeynmanDataset, cf_dicta)
            exp_logger = logging.getLogger(f'{run_name}_Feynman_{size}')
            exp_logger.setLevel(logging.INFO)
            cf_dir = f"{run_path}\Feynman\{size}"
            os.makedirs(cf_dir, exist_ok=True)
            ffh = logging.FileHandler(f"{cf_dir}\log.txt")
            exp_logger.addHandler(ffh)
            ffh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S'))
            
            idx = 1
            for x, y, z, xu, yu, r in tqdm(cf_train_loader):
                eq_st = time.time()
                npx = np.asarray(x)
                npx = npx.reshape(npx.shape[1], npx.shape[0])
                npy = np.asarray(y)
                npy = npy.reshape(npy.shape[0], npy.shape[1])
                model = PySRRegressor(**operator_library, verbosity=0, output_directory=cf_dir, 
                                      run_id = f'{size}_{idx}',
                                      random_state=config['seed'], deterministic=True, parallelism="serial")
                model.fit(npx, npy)
                eq_et = time.time()
                exp_logger.info(f'Proposed equation for {z}:\n {model.sympy()}\n in {eq_et - eq_st} seconds.')
                idx += 1
            
            del exp_logger
            eet = time.time()
            run_logger.info(f'Finished Feynman_{size} run in {eet - sst} seconds.')
        et = time.time()
        run_logger.info(f'Finished Feynman Dataset runs in {et - st} seconds.')

        if run_feynman_units:
            run_logger.info('Initiating Feynman with units Dataset runs.')
            st = time.time()
            for size in sample_sizes:
                sst = time.time()
                run_logger.info(f'Initiating Feynman_units_{size} run.')
                operator_library = {
                    'unary_operators':['sin', 'cos', 'log', 'exp', 'sqrt'], 
                    'binary_operators': ['+', '-', '*', '/', '^']
                    }
                cfu_dicta = {'dataset_dir':config['baseline']['cf_dataset_dir'] + f'/Feynman_{size}',
                            'csv_path':config['baseline']['csv_path'], 
                            'use_units':True,
                            'units_path':config['baseline']['units_path'],
                            'seed':config['seed']
                            }
                cfu_train_loader = get_dataloader(CompressedFeynmanDataset, cfu_dicta)
                exp_logger = logging.getLogger(f'{run_name}_Feynman_{size}')
                exp_logger.setLevel(logging.INFO)
                cfu_dir = f"{run_path}\Feynman_units\{size}"
                os.makedirs(cfu_dir, exist_ok=True)
                ffh = logging.FileHandler(f"{cfu_dir}\log.txt")
                exp_logger.addHandler(ffh)
                ffh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                                   datefmt='%Y-%m-%d %H:%M:%S'))
                
                idx = 1
                for x, y, z, xu, yu, r in tqdm(cfu_train_loader):
                    eq_st = time.time()
                    npx = np.asarray(x)
                    npx = npx.reshape(npx.shape[1], npx.shape[0])
                    npy = np.asarray(y)
                    npy = npy.reshape(npy.shape[0], npy.shape[1])
                    model = PySRRegressor(**operator_library, verbosity=0, output_directory=cfu_dir, 
                                          run_id = f'{size}_{idx}',
                                          random_state=config['seed'], deterministic=True, parallelism="serial")
                    helper_xu(xu)
                    helper_yu(yu)
                    X_units = None
                    y_units = None
                    if xu:
                        X_units = [list(u.values())[0] for u in xu]
                        if all(_ is None for _ in X_units):
                            X_units = None
                        else:
                            for i in range(len(X_units)):
                                if X_units[i] is None:
                                    X_units[i] = ''
                                else:
                                    X_units[i] = X_units[i][0]
                        exp_logger.info(f'Using units: {X_units}')
                    if yu:
                        y_units = list(yu.values())[0]
                        if X_units is not None:
                            if y_units is None:
                                y_units = ''
                                exp_logger.info(f'Using target units: {y_units}')
                    model.fit(npx, npy, X_units=X_units, y_units=y_units)
                    eq_et = time.time()
                    exp_logger.info(f'Proposed equation for {z}: \n {model.sympy()} \n in {eq_et - eq_st} seconds.')
                    idx += 1

                del exp_logger
                eet = time.time()
                run_logger.info(f'Finished Feynman_units_{size} run in {eet - sst} seconds.')
            et = time.time()
            run_logger.info(f'Finished Feynman with units Dataset runs in {et - st} seconds.')
    
    a_et = time.time()
    run_logger.info(f'Finished baseline PySR runs in {a_et - a_st} seconds.')
    return
        