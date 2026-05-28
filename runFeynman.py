import juliacall
from omegaconf import OmegaConf
from datasets.base import get_dataloader
from datasets.feynman_dataset import CompressedFeynmanDataset
from datasets.feynman_dataset import get_compressed_points
from datasets.feynman_dataset import helper_xu
from datasets.feynman_dataset import helper_yu
import numpy as np
import logging
from tqdm import tqdm
from trainers.base import Trainer
import time
import os
import sympy
import gc


def run_Feynman():
    '''
    Runs experiment on FeynmanDataset.
    -------
    '''
    config = OmegaConf.load(PATH)
    if config['main']['run_feynman'] == False:
        return
    idx = 1

    run_name = config['experiments']['Feynman_run_name']
    run_path = os.path.join(config['run']['run_dir'], run_name)
    os.makedirs(run_path, exist_ok=True)

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

    try:
        run_logger.info(f'Initiating experiment {run_name}.')
        cfu_dicta = dict(config['experiments']['Feynman'])
        cfu_train_loader = get_dataloader(CompressedFeynmanDataset, cfu_dicta)
            
        s_t = time.time()
        
        for x, y, eq, xu, yu, r in tqdm(cfu_train_loader):
            npx = np.asarray(x)
            npx = npx.reshape(npx.shape[0], npx.shape[1])
            npy = np.asarray(y)
            npy = npy.reshape(npy.shape[0], npy.shape[1])
            helper_xu(xu)
            helper_yu(yu)
        
            trainer = Trainer(config, npx, npy, eq[0], f'{run_name}\{idx}', dom=r, X_units=xu, y_units=yu)
            
            run_logger.info(f'Initiating train sequence for {eq}')
            start_time = time.time()
            a = trainer.run()
        
            for _ in a:
                    run_logger.info(_)
        
            end_time = time.time()
            run_logger.info(f'finished run {idx}, took {end_time - start_time} seconds')
            print(f'finished run, took {end_time - start_time} seconds')
            
            idx += 1
    
            try: 
                eq = trainer.generator.sympy()
                run_logger.info(f'{eq}')
                run_logger.info(f'{sympy.simplify(eq)}')
                run_logger.info(f'{sympy.symplify(sympy.expand(eq))}')
                del trainer
                gc.collect()
    
            except:
                pass
            
            finally:
                run_logger.info('\n\n')
        
        e_t = time.time()
        run_logger.info(f'finished experiment {run_name}, took {e_t - s_t} seconds')
    except KeyboardInterrupt:
        run_logger.info(f"\n{run_name} experiment interrupted by user.")
    finally:
        return