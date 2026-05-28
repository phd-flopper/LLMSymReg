import juliacall
from omegaconf import OmegaConf
from datasets.base import get_dataloader
from datasets.nguyen_dataset import NguyenDataset
import numpy as np
import logging
from tqdm import tqdm
from trainers.base import Trainer
import time
import os
import sympy
import gc


def run_Nguyen():
    '''
    Runs experiment on NguyenDataset.
    -------
    '''
    config = OmegaConf.load(PATH)
    if config['main']['run_nguyen'] == False:
        return
    idx = 1

    run_name = config['experiments']['Nguyen_run_name']
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
        n_dicta = dict(config['experiments']['Nguyen'])
        n_train_loader = get_dataloader(NguyenDataset, n_dicta, is_train=False)
            
        s_t = time.time()
        
        for x, y, eq in tqdm(n_train_loader):
            npx = np.asarray(x)
            npx = np.vstack(npx).T
            npy = np.asarray(y)
            npy = npy.reshape(npy.shape[-1], npy.shape[0])
            eq = eq[0]
            X = npx
            y = npy
        
            trainer = Trainer(config, X, y, eq, f'{run_name}\{eq}', dom=config['experiments']['dom'][idx - 1])
            
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