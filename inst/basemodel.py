from tqdm import tqdm
from utils import get_length, make_chat_template, auto_log_process
import pandas as pd

@auto_log_process
class preprocess:
    dataset_path: str
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    def process_datasets(self) -> dict:
        df_dicts = {}
        return df_dicts
        
    def make_training_sample(self):
        dicts =self.process_datasets()
        data_dfs = list(dicts.copy().values())
        for i, dataset in enumerate(data_dfs):
            if 'system' not in dataset:
                dataset['system'] = ""
            if 'input' not in dataset:
                dataset['input'] = ""
            if 'chat_template' in dataset:
                continue
            dataset['input'] = dataset['input'].fillna('')
            dataset['system'] = dataset['system'].fillna('')
            dataset['chat_template'] = dataset.apply(lambda x: make_chat_template(x['instruction'], x['input'], x['output'], x['system']),axis=1)
            dataset['length'] = dataset['chat_template'].map(lambda x: get_length(x))
            data_dfs[i] = dataset
            
        data = pd.concat(data_dfs, ignore_index=True)
        data['input'] = data['input'].fillna('')
        data['system'] = data['system'].fillna('')
        return data
        
        