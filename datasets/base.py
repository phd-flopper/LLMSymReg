from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from time import time
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, seed:int=int(time())) -> None:
        '''
        Parameters
        ----------
        int `seed` --- initialize for reproducibility
        '''
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, key:int) -> list:
        raise NotImplementedError


def get_dataloader(
        dataset:BaseDataset, data_config:dict, batch_size:int=1, num_workers:int=0, is_train:bool=False
    ) -> DataLoader:
    '''
    Returns a DataLoader object based on the given config and dataset
    Parameters
    ----------
    Dataset `dataset` --- dataset type
    dict `data_config` --- the config for Dataset object
    int `batch_size` --- the desired batch size (default is 1)
    int `num_workers` --- the desired num_workers value (default is 0)
    bool `is_train` --- whether the output dataset is used for training (default is True)
    '''
    return DataLoader(
        dataset=dataset(**data_config),
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
    )