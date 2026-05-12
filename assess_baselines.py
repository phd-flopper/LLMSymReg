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

def run_all(run_name):

    run_path = os.path.join(config['run']['run_dir'], run_name)
    os.makedirs(run_path, exist_ok=True)
    
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
    run_logger.info('Initiating Nguyen Dataset runs.')
    st = time.time()
    i_size = 0
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
        operator_library = {'maxsize': 30,
                            'maxdepth': 10,
                            'binary_operators': ["+", "-", "*", "/"],
                            'unary_operators':["cos", "exp", "sin", "sqrt", "log"],
                            'constraints':{'pow': (-1, 5), 'log': 5, 'sin': 5, 'cos': 5, 'sqrt': 5},
                            'nested_constraints':{
                                'log': {'log': 0, 'exp': 0}, 
                                'sin': {'sin': 0, 'cos': 0}, 
                                'cos': {'cos': 0, 'sin': 0},
                                'sqrt': {'sqrt': 0},
                                'exp': {'log': 0, 'exp': 0}
                                }
                            }
        if config['baseline']['constrained'] == False:
            operator_library = {
                'unary_operators':['sin', 'cos', 'log', 'exp'], 
                'binary_operators': ['+', '-', '*', '/']
            }
        ext = {'binary_operators': ['+', '-', '*', '/', '^']}
        idx = 1
        n_dicta = {'seed':int(time.time()), 'lib':lib}
        n_train_loader = get_dataloader(NguyenDataset, n_dicta)
        exp_logger = logging.getLogger(f'{run_name}_Nguyen_{size}')
        exp_logger.setLevel(logging.INFO)
        nguyen_dir = f"{run_path}\DNguyen\{size}"
        os.makedirs(nguyen_dir, exist_ok=True)
        ffh = logging.FileHandler(f"{nguyen_dir}\log.txt")
        exp_logger.addHandler(ffh)
        ffh.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S'))
        
        scores = []
        out_scores = []
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
                                      run_id = f'{size}_{eq}', niterations=40)
            model.fit(X, y)
            test_X = n_train_loader.dataset.rng.uniform(*lib[idx])
            y_pred = model.predict(test_X)
            if idx < 9:
                test_X = [test_X[:, 0]]
            else:
                test_X = test_X[:, 0], test_X[:, 1]
            test_y = NGUYEN_EQUATIONS[eq](*test_X)
            try:
                score = r2_score(test_y, y_pred)
            except:
                score = 0
            scores.append(score)
            out_dom = lib[idx]
            out_dom[0] = out_dom[0] * 10
            out_dom[1] = out_dom[1] * 10
            out_test_X = n_train_loader.dataset.rng.uniform(*out_dom)
            out_y_pred = model.predict(out_test_X)
            if idx < 9:
                out_test_X = [out_test_X[:, 0]]
            else:
                out_test_X = out_test_X[:, 0], out_test_X[:, 1]
            out_test_y = NGUYEN_EQUATIONS[eq](*out_test_X)
            try:
                out_score = r2_score(out_test_y, out_y_pred)
            except:
                out_score = 0
            out_scores.append(out_score)
            idx += 1
            eq_et = time.time()
            equations[i_size][eq].append(model.sympy())
            exp_logger.info(f'Proposed equation for {eq}:\n {model.sympy()} \n in {eq_et - eq_st} seconds.')
            exp_logger.info(f'R^2 score: {score}')
            
        scores = np.array(scores)
        scores /= num_runs
        out_scores = np.array(out_scores)
        out_scores /= num_runs
        global_scores[i_size] = np.add(global_scores[i_size], scores)
        out_global_scores[i_size] = np.add(out_global_scores[i_size], out_scores)
        del exp_logger
        i_size += 1
        eet = time.time()
        run_logger.info(f'Finished Nguyen_{size} run in {eet - sst} seconds.')
        run_logger.info(f'R^2 scores: {scores * num_runs}.')
    et = time.time()
    run_logger.info(f'Finished Nguyen Dataset runs in {et - st} seconds.')
    
    a_et = time.time()
    run_logger.info(f'Finished baseline PySR runs in {a_et - a_st} seconds.')
    return

def plot_():
    average_scores = [sum([0 if x < 0 else x for x in g_s]) / len(g_s) for g_s in global_scores]
    neg_fracs = [sum([1 if x < 0 else 0 for x in g_s]) / len(g_s) for g_s in global_scores]
    sample_sizes = [str(_) for _ in config['baseline']['sample_sizes']]
    colors = ['green', 'orange', 'red', 'blue', 'purple']
    colors = colors[:len(sample_sizes)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(sample_sizes))
    
    ax1.bar(x, average_scores, width=0.33, color=colors)
    ax1.set_title('Average $R^2$ score (ID)')
    ax1.set_xlabel('Sample size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_sizes)
    
    ax2.bar(x, neg_fracs, width=0.33, color=colors)
        
    ax2.set_title('Fraction of negative $R^2$ (ID)')
    ax2.set_xlabel('Sample size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sample_sizes)
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
    sample_sizes = [str(_) for _ in config['baseline']['sample_sizes']]
    colors = ['green', 'orange', 'red', 'blue', 'purple']
    colors = colors[:len(sample_sizes)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(sample_sizes))
    
    ax1.bar(x, average_scores, width=0.33, color=colors)
    ax1.set_title('Average $R^2$ score (OOD)')
    ax1.set_xlabel('Sample size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_sizes)
    
    ax2.bar(x, neg_fracs, width=0.33, color=colors)
        
    ax2.set_title('Fraction of negative $R^2$ (OOD)')
    ax2.set_xlabel('Sample size')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sample_sizes)
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
    global_scores = [[0] * 12 for _ in range(len(config['baseline']['sample_sizes']))]
    out_global_scores = [[0] * 12 for _ in range(len(config['baseline']['sample_sizes']))]
    
    num_runs = config['baseline']['num_runs']
    
    equations = [{f'Nguyen_{i}': [] for i in range(1, 13)} for _ in range(len(config['baseline']['sample_sizes']))]
    
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
    
    for j in range(len(config['baseline']['sample_sizes'])):
        for i in range(1, 13):
            eq = f'Nguyen_{i}'
            with open(f_path + f'\{config["baseline"]["sample_sizes"][j]}_{eq}.txt', 'w+') as f:
                for line in equations[j][eq]:
                    f.write(str(line) + '\n')
        f.close()
    
    
    plot_()
    plot_out_()