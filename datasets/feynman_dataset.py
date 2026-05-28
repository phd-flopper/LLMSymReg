from datasets.base import BaseDataset
import numpy as np
import pandas as pd
import os
import linecache
import gc
from tqdm import tqdm
from time import time

def fix_csv(csv_path:str) -> None:
    '''
    FeynmanEquations.csv inherently has a few broken rows; this helper function fixes the .csv file inplace.
    '''
    df = pd.read_csv(csv_path)
    try: 
        df.loc[df.Filename == 'I.15.1', 'Filename'] = 'I.15.10'
    except:
        pass
    try: 
        df.loc[df.Filename == 'I.48.2', 'Filename'] = 'I.48.20'
    except:
        pass
    try: 
        df.loc[df.Filename == 'II.11.17', 'Filename'] = 'II.11.7'
    except:
        pass
    try: 
        df.loc[df.Filename == 'II.37.1', '# variables'] = 3
    except:
        pass
    df.to_csv(csv_path)


def get_compressed_points(output_dir:str, dataset_dir:str, csv_file:str, 
                          sample_size:int, seed:int=int(time())) -> None:
    '''
    Hepler function to prepare compressed files for CompressedFeynmanDataset.
    /run this before working with CompressedFeynmanDataset instances/
    Parameters
    ----------
    str `output_dir` --- path to the output directory
    str `dataset_dir` --- path to original files
    str `csv_file` --- path to the file of equations dataframe with metadata
    int `sample_size` --- how many points are sampled for each equation
    int `seed` --- initialize for reproducibility
    '''
    df = pd.read_csv(csv_file).head(100)
    rng = np.random.default_rng(seed=seed)
    output_dir += f'/Feynman_{sample_size}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in tqdm(range(1, 101)):
        fname = df[df['Number'] == i]['Filename'].to_list()[0]
        fpath = dataset_dir + f'/{fname}'
        with open(fpath) as f:
            high = len(f.readlines())
            f.close()
        ids = rng.integers(low=0, high=high, size=sample_size)
        del high
        gc.collect()
        ids.sort()
        points = [linecache.getline(fpath, _) for _ in ids]
        with open(output_dir + f'/{fname}', 'w+') as f:
            for line in points:
                f.write(line)
            f.close()
        linecache.clearcache()
    print('Successfully created compressed files.')


class CompressedFeynmanDataset(BaseDataset):
    def __init__(
            self, 
            dataset_dir:str, 
            csv_path:str, 
            seed:int=int(time()),
            n_equations:int=100,
            subset:list=None, 
            use_units:bool=False, 
            units_path:str=None,
            create_dir:bool=False,
            sample_size:int=None,
            uncompressed_path:str=None
        ) -> None:
        '''
        Feynman equations dataset with points files compressed to specified size.
        Parameters
        ----------
        str `dataset_dir` --- path to dataset root
        str `csv_path` --- path to the file of equations dataframe with metadata
        int `seed` --- initialize for reproducibility
        int `n_equations` --- how many Feynman equations to consider (defaults to 100 i.e. all)
        optional list `subset` --- if specified, only passed Feynman equations are considered
        bool `use_units` --- whether to consider units; defaults to False
        str `units_path` --- filename of units dataframe; specified if `use_units` is True
        bool `create_dir` --- whether to create compressed files via `get_compressed_points` helper function /
        (defaults to False, implying the compressed files already exist)
        int `sample_size` --- how many points are sampled for each equation; specify if `create_dir` is True
        str `uncompressed_path` --- path to uncompressed files directory; specify if `create_dir` is True
        '''
        super(CompressedFeynmanDataset, self).__init__(seed)
        self.df = pd.read_csv(csv_path).head(100)
        self.dataset_dir = dataset_dir
        if create_dir is True:
            assert uncompressed_path is not None, 'please specify `uncompressed_path`'
            get_compressed_points(dataset_dir, uncompressed_path, csv_path, sample_size, seed)
            self.dataset_dir += f'/Feynman_{sample_size}'
        self.n_equations = n_equations
        self.subset = subset
        if self.subset:
            assert len(self.subset) == self.n_equations, '`n_equations` argument must equal `subset` length'
            self.subset.sort()
        self.units_df = None
        self.use_units = use_units
        if self.use_units:
            self.units_df = pd.read_csv(units_path)
        self.X_units = 'None'
        self.y_units = 'None'

    def __len__(self) -> int:
        return self.n_equations

    def __getitem__(self, key:int) -> list:
        eq_num = key + 1
        if self.subset:
            eq_num = self.subset[key]
        num_vars = self.df[self.df['Number'] == eq_num]["# variables"].astype(int).to_list()[0]
        fname = self.df[self.df['Number'] == eq_num]['Filename'].to_list()[0]
        formula = self.df[self.df['Number'] == eq_num]['Formula'].to_list()[0]
        self.range = self.get_range(eq_num, num_vars)
        self.X_units, self.y_units = self.get_units(eq_num, num_vars)
        with open(self.dataset_dir + f'/{fname}') as f:
            self.points = [[float(x) for x in k.split()] for k in f.readlines()]
            f.close()
        X = [[vec[_] for _ in range(num_vars)] for vec in self.points]
        y = [vec[-1] for vec in self.points]
        return X, y, formula, self.X_units, self.y_units, self.range

    def get_units(self, eq_num:int, num_vars:int) -> list:
        if self.use_units == False:
            return ['None', 'None']
        df_row = self.df[self.df['Number'] == eq_num]
        target = df_row['Output'].to_list()[0]
        tmp = self.units_df[self.units_df['Variable'] == target]
        y_out = {tmp['Units'].to_list()[0]:self.parse_units(tmp)}
        x_cols = [f'v{i+1}_name' for i in range(num_vars)]
        x_out = []
        for col_name, data in df_row[x_cols].items():
            var = data.values[0]
            tmp = self.units_df[self.units_df['Variable'] == var]
            x_out.append({tmp['Units'].to_list()[0]:self.parse_units(tmp)})
        return x_out, y_out

    def parse_units(self, row:pd.core.frame.DataFrame) -> str:
        res = ''
        for col_name, data in row[["m", "s", "kg", "T", "V"]].items():
            val = data.values.astype(int)[0]
            if val == 0:
                continue
            elif val == 1:
                if res != '':
                    res += ' * '
                res += col_name
            elif val > 1:
                if res != '':
                    res += ' * '
                res += f'{col_name}^{val}'
            elif val == -1:
                if res != '':
                    res += ' / '
                else:
                    res += '1 / '
                res += f'{col_name}'
            elif val < -1:
                if res != '':
                    res += ' / '
                res += f'{col_name}^{-val}'
        if res != '':
            return res
        return 'None'

    def get_range(self, eq_num:int, num_vars:int) -> list:
        df_row = self.df[self.df['Number'] == eq_num]
        out = [[] for _ in range(num_vars)]
        idx = 0
        for col_name, data in df_row[[f'v{i+1}_low' for i in range(num_vars)]].items():
            out[idx].append(data.values[0])
            idx += 1
        idx = 0
        for col_name, data in df_row[[f'v{i+1}_high' for i in range(num_vars)]].items():
            out[idx].append(data.values[0])
            idx += 1
        return out


class FeynmanDataset(BaseDataset):
    def __init__(
            self, 
            dataset_dir:str, 
            csv_file:str, 
            seed:int=int(time()),
            n_equations:int=100,
            sample_size:int=20,
            subset:list=None, 
            use_units:bool=False, 
            units_path:str=None
        ) -> None:
        '''
        Full Feynman equations dataset as is.
        Parameters
        ----------
        str `dataset_dir` --- path to dataset root
        str `csv_file` --- filename of equations dataframe with metadata
        int `seed` --- initialize for reproducibility
        int `n_equations` --- how many Feynman equations to consider (defaults to 100 i.e. all)
        int `sample_size` --- number of points sampled for each variable
        optional list `subset` --- if specified, only passed Feynman equations are considered
        bool `use_units` --- whether to consider units; defaults to False
        str `units_path` --- filename of units dataframe; specified if `use_units` is True
        '''
        super(FeynmanDataset, self).__init__(seed)
        self.df = pd.read_csv(os.path.join(dataset_dir, csv_file)).head(100)
        self.n_equations = n_equations
        self.sample_size = sample_size
        self.subset = subset
        if self.subset:
            assert len(self.subset) == self.n_equations, '`n_equations` argument must equal `subset` length'
            self.subset.sort()
        self.units_df = None
        self.use_units = use_units
        points_path = '/Feynman_with_units/'
        if self.use_units:
            self.units_df = pd.read_csv(units_path)
        self.points_path = dataset_dir + points_path
        self.X_units = 'None'
        self.y_units = 'None'

    def __len__(self) -> int:
        return self.n_equations

    def __getitem__(self, key:int) -> list:
        eq_num = key + 1
        if self.subset:
            eq_num = self.subset[key]
        num_vars = self.df[self.df['Number'] == eq_num]["# variables"].astype(int).to_list()[0]
        fname = self.df[self.df['Number'] == eq_num]['Filename'].to_list()[0]
        formula = self.df[self.df['Number'] == eq_num]['Formula'].to_list()[0]
        self.range = self.get_range(eq_num, num_vars)
        self.X_units, self.y_units = self.get_units(eq_num, num_vars)
        high = 0
        fpath = os.path.join(self.points_path, fname)
        with open(fpath) as f:
            high = len(f.readlines())
            f.close()
        ids = self.rng.integers(low=0, high=high, size=self.sample_size)
        del high
        gc.collect()
        ids.sort()
        self.points = [[float(x) for x in k.split()] for k in [linecache.getline(fpath, _) for _ in ids]]
        linecache.clearcache()
        X = [[vec[_] for _ in range(num_vars)] for vec in self.points]
        y = [vec[-1] for vec in self.points]
        return X, y, formula, self.X_units, self.y_units, self.range

    def get_units(self, eq_num:int, num_vars:int) -> list:
        if self.use_units == False:
            return ['None', 'None']
        df_row = self.df[self.df['Number'] == eq_num]
        target = df_row['Output'].to_list()[0]
        tmp = self.units_df[self.units_df['Variable'] == target]
        y_out = {tmp['Units'].to_list()[0]:self.parse_units(tmp)}
        x_cols = [f'v{i+1}_name' for i in range(num_vars)]
        x_out = []
        for col_name, data in df_row[x_cols].items():
            var = data.values[0]
            tmp = self.units_df[self.units_df['Variable'] == var]
            x_out.append({tmp['Units'].to_list()[0]:self.parse_units(tmp)})
        return x_out, y_out

    def parse_units(self, row:pd.core.frame.DataFrame) -> str:
        res = ''
        for col_name, data in row[["m", "s", "kg", "T", "V"]].items():
            val = data.values.astype(int)[0]
            if val == 0:
                continue
            elif val == 1:
                if res != '':
                    res += ' * '
                res += col_name
            elif val > 1:
                if res != '':
                    res += ' * '
                res += f'{col_name}^{val}'
            elif val == -1:
                if res != '':
                    res += ' / '
                else:
                    res += '1 / '
                res += f'{col_name}'
            elif val < -1:
                if res != '':
                    res += ' / '
                res += f'{col_name}^{-val}'
        if res != '':
            return res
        return 'None'
    
    def get_range(self, eq_num:int, num_vars:int) -> list:
        df_row = self.df[self.df['Number'] == eq_num]
        out = [[] for _ in range(num_vars)]
        idx = 0
        for col_name, data in df_row[[f'v{i+1}_low' for i in range(num_vars)]].items():
            out[idx].append(data.values[0])
            idx += 1
        idx = 0
        for col_name, data in df_row[[f'v{i+1}_high' for i in range(num_vars)]].items():
            out[idx].append(data.values[0])
            idx += 1
        return out


def helper_xu(xu):
    if type(xu) is tuple:
        xu = (None,)
    else:
        for i in xu:
            for _ in i.keys():
                if i[_] == ['None']:
                    i[_] = None


def helper_yu(yu):
    if type(yu) is tuple:
        yu = (None,)
    else:
        for _ in yu.keys():
            if yu[_] == ['None']:
                yu[_] = None
            else:
                yu[_] = yu[_][0]