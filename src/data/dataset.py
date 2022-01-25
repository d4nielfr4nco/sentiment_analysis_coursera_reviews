import datetime
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, 'data')


class Dataset:

    @classmethod
    def sample_dataframe(cls, df, pct):
        if pct in np.linspace(0.1, 100, 1000):
            num_samples = int(len(df) * (pct / 100))
            return df.sample(num_samples, random_state=2145)
        else:
            raise ValueError('Percentage parameter should be between 0.1% and 100%')

    @classmethod
    def save_dataframe(cls, df, file_name, file_path, add_date=True):
        try:
            file_name_only, file_ext = file_name.split('.')
        except ValueError:
            raise ValueError('Expected file extension in file name.')
        if file_ext in ['pkl', 'csv']:
            if add_date:
                current_time = datetime.datetime.utcnow().isoformat().split('.')[0].replace('-','').replace('T','_').replace(':','')
                file_name = current_time + ' - ' + file_name

            file_name = os.path.join(DATA_DIR, file_path, file_name)
            if file_ext == 'csv':
                df.to_csv(file_name)
            else:
                df.to_pickle(file_name)
        else:
            raise ValueError('Supported file extensions: csv and pkl.')

    @classmethod
    def load_dataframe(cls, file_name, file_path):
        try:
            file_name_only, file_ext = file_name.split('.')
        except ValueError:
            raise ValueError('Expected file extension in file name.')
        if file_ext in ['pkl', 'csv']:
            file_name = os.path.join(DATA_DIR, file_path, file_name)
            if file_ext == 'csv':
                df = pd.read_csv(file_name)
            else:
                df = pd.read_pickle(file_name)
            return df
        else:
            raise ValueError('Supported file extensions: csv and pkl.')

    @classmethod
    def save_vect_corpus(cls, X, y, file_name, file_path, add_date=True):
        try:
            file_name_only, file_ext = file_name.split('.')
        except ValueError:
            raise ValueError('Expected file extension in file name.')
        if file_ext == 'npy':
            if add_date:
                current_time = datetime.datetime.utcnow().isoformat().split('.')[0].replace('-','').replace('T','_').replace(':','')
                file_name = current_time + ' - ' + file_name_only

            X_file_name = os.path.join(DATA_DIR, file_path, file_name + 'X.' + file_ext)
            y_file_name = os.path.join(DATA_DIR, file_path, file_name + 'y.' + file_ext)
            np.save(file=X_file_name, arr=X)
            np.save(file=y_file_name, arr=y)
        else:
            raise ValueError('Supported file extension: pkl.')

    @classmethod
    def load_vect_corpus(cls, X_file_name, y_file_name, file_path):
        try:
            X_file_name_only, X_file_ext = X_file_name.split('.')
            y_file_name_only, y_file_ext = y_file_name.split('.')
        except ValueError:
            raise ValueError('Expected file extension in file name.')
        if X_file_ext == 'npy' and y_file_ext == 'pkl':
            X_file_name = os.path.join(DATA_DIR, file_path, X_file_name)
            y_file_name = os.path.join(DATA_DIR, file_path, y_file_name)
            return np.load(X_file_name), np.load(y_file_name)
        else:
            raise ValueError('Supported file extension: pkl.')

    @classmethod
    def split_dataset(cls, X, y, test_size):
        return train_test_split(X, y, test_size=test_size, random_state=57, stratify=y)
