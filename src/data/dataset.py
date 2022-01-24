import datetime
import os
import pandas as pd
import numpy as np
from pathlib import Path

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
