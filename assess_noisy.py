import juliacall
from pysr import PySRRegressor
from omegaconf import OmegaConf
from datasets.base import get_dataloader
from datasets.nguyen_dataset import NguyenDataset
import numpy as np
from tqdm import tqdm
import logging
import time
import os
from sklearn.metrics import r2_score
from datasets.nguyen_dataset import NGUYEN_EQUATIONS
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from trainers.noisy import Trainer
import sympy

def run_all(run_name):

    run_path = os.path.join(config['run']['run_dir'], run_name)
    os.makedirs(run_path, exist_ok=True)
    
    levels = config['noisy']['levels']
    
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
    
    run_logger.info('Initiating model runs.')
    a_st = time.time()
    run_logger.info('Initiating Nguyen Dataset runs.')
    st = time.time()
    i_size = 0
    for level in tqdm(levels):
        sst = time.time()
        run_logger.info(f'Initiating Nguyen with noise level {int(level * 100)} run.')
        lib = dict(config['noisy']['lib'])
        idx = 1
        n_dicta = {'seed':int(time.time()), 'lib':lib}
        n_train_loader = get_dataloader(NguyenDataset, n_dicta)
        exp_logger = logging.getLogger(f'{run_name}_Nguyen_{int(level * 100)}')
        exp_logger.setLevel(logging.INFO)
        nguyen_dir = f"{run_path}\DNguyen\{int(level * 100)}"
        os.makedirs(nguyen_dir, exist_ok=True)
        ffh = logging.FileHandler(f"{nguyen_dir}\log.txt")
        exp_logger.addHandler(ffh)
        ffh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S'))
        
        scores = []
        out_scores = []
        for x, y, eq in tqdm(n_train_loader):
            eq_st = time.time()
            npx = np.asarray(x)
            npx = np.vstack(npx).T
            npy = np.asarray(y)
            npy = npy.reshape(npy.shape[-1], npy.shape[0])
            eq = eq[0]
            X = npx
            y = npy
            
            noise = np.random.normal(loc=0.0, scale=level*np.abs(y[:20, 0]), size=20)
            
            for i in range(X.shape[1]): 
                X[:20, i] += noise

            y[:20, 0] += noise
            
            trainer = Trainer(config, X, y, eq, f'{int(level * 100)}_{eq}', dom=config['experiments']['dom'][idx - 1])
            trainer.run()
            test_X = n_train_loader.dataset.rng.uniform(*lib[idx])
            if idx < 9:
                test_X = [test_X[:, 0]]
            else:
                test_X = test_X[:, 0], test_X[:, 1]
            try:
                score = max(trainer.scores)
            except:
                score = 0
            scores.append(score)
            out_dom = lib[idx]
            out_dom[0] = out_dom[0] * 10
            out_dom[1] = out_dom[1] * 10
            out_test_X = n_train_loader.dataset.rng.uniform(*out_dom)
            if trainer.best_generator is not None:
                out_y_pred = trainer.best_generator.predict(out_test_X)
                if idx < 9:
                    out_test_X = [out_test_X[:, 0]]
                else:
                    out_test_X = out_test_X[:, 0], out_test_X[:, 1]
                out_test_y = NGUYEN_EQUATIONS[eq](*out_test_X)
                try:
                    out_score = r2_score(out_test_y, out_y_pred)
                except:
                    out_score = 0
                if np.isnan(out_score):
                    out_score = 0
                for pair in trainer.mutator_hof:
                    try: 
                        func = pair['equation']
                        func = func.replace("^", "**")
                        function = sympy.lambdify(trainer.vars_, func)
                        function_score = r2_score(out_test_y, function(*out_test_X))
                        out_score = max(out_score, function_score)
                    except:
                        continue
                out_scores.append(out_score)
                idx += 1
                eq_et = time.time()
                equations[i_size][eq].append(trainer.best_generator.sympy())
                exp_logger.info(f'Proposed equation for {eq}:\n {trainer.best_generator.sympy()} \n in {eq_et - eq_st} seconds.')
                exp_logger.info(f'R^2 score: {score}')
            else:
                out_score = 0
                exp_logger.info('Something went wrong')
                out_scores.append(out_score)
                idx += 1
                eq_et = time.time()
                exp_logger.info('R^2 score: 0')
            
        scores = np.array(scores)
        scores /= num_runs
        out_scores = np.array(out_scores)
        out_scores /= num_runs
        global_scores[i_size] = np.add(global_scores[i_size], scores)
        out_global_scores[i_size] = np.add(out_global_scores[i_size], out_scores)
        del exp_logger
        i_size += 1
        eet = time.time()
        run_logger.info(f'Finished Nguyen with noise level {int(level * 100)} run in {eet - sst} seconds.')
        run_logger.info(f'R^2 scores: {scores * num_runs}.')
    et = time.time()
    run_logger.info(f'Finished Nguyen Dataset runs in {et - st} seconds.')
    
    a_et = time.time()
    run_logger.info(f'Finished baseline PySR runs in {a_et - a_st} seconds.')
    return

def plot_():
    average_scores = [sum([0 if x < 0 else x for x in g_s]) / len(g_s) for g_s in global_scores]
    neg_fracs = [sum([1 if x < 0 else 0 for x in g_s]) / len(g_s) for g_s in global_scores]
    levels = [str(int(_ * 100)) for _ in config['noisy']['levels']]
    colors = ['lime', 'orange', 'red', 'blue', 'purple', 'brown', 'pink', 'gray', 'green', 'olive', 'cyan']
    colors = colors[:len(levels)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(levels))
    
    ax1.bar(x, average_scores, width=0.33, color=colors)
    ax1.set_title('Average $R^2$ score (ID)')
    ax1.set_xlabel('Noise level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    
    ax2.bar(x, neg_fracs, width=0.33, color=colors)
        
    ax2.set_title('Fraction of negative $R^2$ (ID)')
    ax2.set_xlabel('Noise level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    
    plt.tight_layout()
    plt.savefig(config['run']['run_dir'] + '\scores.svg')
    plt.savefig(config['run']['run_dir'] + '\scores')
    plt.show()

def plot_out_():
    average_scores = [sum([0 if x < 0 else x for x in g_s]) / len(g_s) for g_s in out_global_scores]
    neg_fracs = [sum([1 if x < 0 else 0 for x in g_s]) / len(g_s) for g_s in out_global_scores]
    levels = [str(int(_ * 100)) for _ in config['noisy']['levels']]
    colors = ['lime', 'orange', 'red', 'blue', 'purple', 'brown', 'pink', 'gray', 'green', 'olive', 'cyan']
    colors = colors[:len(levels)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(levels))
    
    ax1.bar(x, average_scores, width=0.33, color=colors)
    ax1.set_title('Average $R^2$ score (OOD)')
    ax1.set_xlabel('Noise level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    
    ax2.bar(x, neg_fracs, width=0.33, color=colors)
        
    ax2.set_title('Fraction of negative $R^2$ (OOD)')
    ax2.set_xlabel('Noise level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
    
    plt.tight_layout()
    plt.savefig(config['run']['run_dir'] + '\out_scores.svg')
    plt.savefig(config['run']['run_dir'] + '\out_scores')
    plt.show()

if __name__ == "__main__":
    config = OmegaConf.load(PATH)
    global_scores = [[0] * 12 for _ in range(len(config['noisy']['levels']))]
    out_global_scores = [[0] * 12 for _ in range(len(config['noisy']['levels']))]
    
    num_runs = config['noisy']['num_runs']
    
    equations = [{f'Nguyen_{i}': [] for i in range(1, 13)} for _ in range(len(config['noisy']['levels']))]
    
    for i in range(1, num_runs + 1):
        run_all(f'Trial_Nguyen_{i}')
    
    with open(config['run']['run_dir'] + '\scores.txt', 'w+') as f:
        for line in global_scores:
            f.write(str(line) + '\n')
        f.close()

    np.save(config['run']['run_dir'] + '\global_scores.npy', global_scores)  

    with open(config['run']['run_dir'] + '\out_scores.txt', 'w+') as f:
        for line in out_global_scores:
            f.write(str(line) + '\n')
        f.close()
        
    np.save(config['run']['run_dir'] + '\out_global_scores.npy', out_global_scores)
    
    f_path = config['run']['run_dir'] + '\equations'
    os.makedirs(f_path, exist_ok=True)
    
    for j in range(len(config['noisy']['levels'])):
        for i in range(1, 13):
            eq = f'Nguyen_{i}'
            with open(f_path + f'\{int(config["noisy"]["levels"][j] * 100)}_{eq}.txt', 'w+') as f:
                for line in equations[j][eq]:
                    f.write(str(line) + '\n')
        f.close()
    
    
    plot_()
    plot_out_()