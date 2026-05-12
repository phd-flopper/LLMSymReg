from datasets.base import BaseDataset
from datasets.base import get_dataloader
from datasets.nguyen_dataset import NguyenDataset
from datasets.feynman_dataset import FeynmanDataset
from datasets.feynman_dataset import CompressedFeynmanDataset
from datasets.feynman_dataset import get_compressed_points
from datasets.feynman_dataset import helper_xu
from datasets.feynman_dataset import helper_yu
import numpy as np
from tqdm import tqdm
import time
from omegaconf import OmegaConf


config = OmegaConf.load(PATH)

lib = {
       1:[-1, 1, [20, 1]],
       2:[-1, 1, [20, 1]],
       3:[-1, 1, [20, 1]],
       4:[-1, 1, [20, 1]],
       5:[-1, 1, [20, 1]],
       6:[-1, 1, [20, 1]],
       7:[0, 2, [20, 1]],
       8:[0, 4, [20, 1]],
       9:[0, 1, [20, 2]],
       10:[0, 1, [20, 2]],
       11:[0, 1, [20, 2]],
       12:[0, 1, [20, 2]],
}

n_dicta = {'seed':42, 'lib':lib, 'n_equations':12, 'subset':None}
start_time = time.time()
n_train_loader = get_dataloader(NguyenDataset, n_dicta, is_train=False)
end_time = time.time()
print(f'DataLoader object for NguyenDataset successfully built, took {end_time - start_time} seconds.')

start_time = time.time()
for x, y, z in tqdm(n_train_loader):
    print(f'{z[0]} equation successfully built.')
    npx = np.asarray(x)
    npx = npx.reshape(npx.shape[-1], npx.shape[0])
    npy = np.asarray(y)
    npy = npy.reshape(npy.shape[-1], npy.shape[0])
end_time = time.time()
print(f'Successfully iterated Dataloader, took {end_time - start_time} seconds.')



print('Testing `get_compressed_points` helper function.')
start_time = time.time()
get_compressed_points(
    output_dir=config['gcp']['output_dir'], 
    dataset_dir=config['gcp']['dataset_dir'], 
    csv_file=config['gcp']['csv_file'], 
    sample_size=config['gcp']['sample_size']
)
end_time = time.time()
print(f'Successfully sampled files, took {end_time - start_time} seconds.')



f_dicta = {
    'dataset_dir':config['f']['dataset_dir'], 
    'csv_file':config['f']['csv_file'], 
    'seed':config['f']['seed'], 
    'n_equations':config['f']['n_equations'],
    'sample_size':config['f']['sample_size']
    }
start_time = time.time()
f_train_loader = get_dataloader(FeynmanDataset, f_dicta, is_train=False)
end_time = time.time()
print(f'DataLoader object for FeynmanDataset (without units) successfully built, took {end_time - start_time} seconds.')

last = None
start_time = time.time()
for x, y, z, xu, yu, r in tqdm(f_train_loader):
    npx = np.asarray(x)
    npx = npx.reshape(npx.shape[1], npx.shape[0])
    npy = np.asarray(y)
    npy = npy.reshape(npy.shape[0], npy.shape[1])
    helper_xu(xu)
    helper_yu(yu)
    last = [npx, npy, z, xu, yu, r]
for l in last:
    print(l)
end_time = time.time()
print(f'Successfully iterated Dataloader, took {end_time - start_time} seconds.')



fu_dicta = {
    'dataset_dir':config['fu']['dataset_dir'], 
    'csv_file':config['fu']['csv_file'], 
    'seed':config['fu']['seed'], 
    'n_equations':config['fu']['n_equations'],
    'sample_size':config['fu']['sample_size'],
    'use_units':config['fu']['use_units'],
    'units_path':config['fu']['units_path']
    }
start_time = time.time()
fu_train_loader = get_dataloader(FeynmanDataset, fu_dicta, is_train=False)
end_time = time.time()
print(f'DataLoader object for FeynmanDataset (with units) successfully built, took {end_time - start_time} seconds.')

last = None
start_time = time.time()
for x, y, z, xu, yu, r in tqdm(fu_train_loader):
    npx = np.asarray(x)
    npx = npx.reshape(npx.shape[1], npx.shape[0])
    npy = np.asarray(y)
    npy = npy.reshape(npy.shape[0], npy.shape[1])
    if type(xu) is tuple:
        xu = (None,)
    else:
        for i in xu:
            for _ in i.keys():
                if i[_] == ['None']:
                    i[_] = None
    if type(yu) is tuple:
        yu = (None,)
    else:
        for _ in yu.keys():
            if yu[_] == ['None']:
                yu[_] = None
            else:
                yu[_] = yu[_][0]
    last = [npx, npy, z, xu, yu, r]
for l in last:
    print(l)
end_time = time.time()
print(f'Successfully iterated Dataloader, took {end_time - start_time} seconds.')



cf_dicta = {
    'dataset_dir':config['cf']['dataset_dir'], 
    'csv_path':config['cf']['csv_path'], 
    'seed':config['cf']['seed'],
    'n_equations':config['cf']['n_equations'],
    'create_dir':config['cf']['create_dir'], 
    'sample_size':config['cf']['sample_size'],
    'uncompressed_path':config['cf']['uncompressed_path'],
    }
start_time = time.time()
cf_train_loader = get_dataloader(CompressedFeynmanDataset, cf_dicta, is_train=False)
end_time = time.time()
print(f'DataLoader object for CompressedFeynmanDataset (without units, with sampling files) successfully built, took {end_time - start_time} seconds.')

last = None
start_time = time.time()
for x, y, z, xu, yu, r in tqdm(cf_train_loader):
    npx = np.asarray(x)
    npx = npx.reshape(npx.shape[1], npx.shape[0])
    npy = np.asarray(y)
    npy = npy.reshape(npy.shape[0], npy.shape[1])
    helper_xu(xu)
    helper_yu(yu)
    last = [npx, npy, z, xu, yu, r]
for l in last:
    print(l)
end_time = time.time()
print(f'Successfully iterated Dataloader, took {end_time - start_time} seconds.')



cfu_dicta = {
    'dataset_dir':config['cfu']['dataset_dir'], 
    'csv_path':config['cfu']['csv_path'], 
    'seed':config['cfu']['seed'], 
    'n_equations':config['cfu']['n_equations'],
    'sample_size':config['cfu']['sample_size'],
    'use_units':config['cfu']['use_units'],
    'units_path':config['cfu']['units_path']
    }
start_time = time.time()
cfu_train_loader = get_dataloader(CompressedFeynmanDataset, cfu_dicta, is_train=False)
end_time = time.time()
print(f'DataLoader object for CompressedFeynmanDataset (with units) successfully built, took {end_time - start_time} seconds.')

last = None
start_time = time.time()
for x, y, z, xu, yu, r in tqdm(cfu_train_loader):
    npx = np.asarray(x)
    npx = npx.reshape(npx.shape[1], npx.shape[0])
    npy = np.asarray(y)
    npy = npy.reshape(npy.shape[0], npy.shape[1])
    helper_xu(xu)
    helper_yu(yu)
    last = [npx, npy, z, xu, yu, r]
for l in last:
    print(l)
end_time = time.time()
print(f'Successfully iterated Dataloader, took {end_time - start_time} seconds.')