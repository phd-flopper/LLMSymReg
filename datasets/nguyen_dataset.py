from datasets.base import BaseDataset
import numpy as np
from time import time

NGUYEN_EQUATIONS = {
    'Nguyen_1': lambda x: x**3 + x**2 + x,
    'Nguyen_2': lambda x: x**4 + x**3 + x**2 + x,
    'Nguyen_3': lambda x: x**5 + x**4 + x**3 + x**2 + x,
    'Nguyen_4': lambda x: x**6 + x**5 + x**4 + x**3 + x**2 + x,
    'Nguyen_5': lambda x: np.sin(x**2) * np.cos(x) - 1,
    'Nguyen_6': lambda x: np.sin(x) + np.sin(x + x**2),
    'Nguyen_7': lambda x: np.log(x + 1) + np.log(x**2 + 1),
    'Nguyen_8': lambda x: np.sqrt(x),
    'Nguyen_9': lambda x, y: np.sin(x) + np.sin(y**2),
    'Nguyen_10': lambda x, y: 2 * np.sin(x) * np.cos(y),
    'Nguyen_11': lambda x, y: x ** y,
    'Nguyen_12': lambda x, y: x ** 4 - x ** 3 + 0.5 * y ** 2 - y  
}

default_lib = {
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


class NguyenDataset(BaseDataset):
    def __init__(
            self, 
            seed:int=int(time()), 
            lib:dict=default_lib, 
            n_equations:int=12, 
            subset:list=None
        ) -> None:
        '''
        Parameters
        ----------
        int `seed` --- initialize for reproducibility
        dict `lib` --- dictionary with low, high and dimension for each equation dataset
        int `n_equations` --- how many Nguyen equations to consider (defaults to 12 i.e. all)
        optional `list subset` --- if specified, only passed Nguyen equations are considered
        '''
        super(NguyenDataset, self).__init__(seed)
        self.lib = lib
        self.n_equations = n_equations
        self.subset = subset
        if self.subset:
            assert len(self.subset) == self.n_equations, '`n_equations` argument must equal `subset` length'
            self.subset.sort()

    def __len__(self) -> int:
        return self.n_equations

    def __getitem__(self, key:int) -> list:
        eq_num = key + 1
        if self.subset:
            eq_num = self.subset[key]
        self.points = self.rng.uniform(*self.lib[eq_num])
        if eq_num < 9:
            self.points = [self.points[:, 0]]
        else:
            self.points = self.points[:, 0], self.points[:, 1]
        return self.points, NGUYEN_EQUATIONS[f'Nguyen_{eq_num}'](*self.points), f'Nguyen_{eq_num}'