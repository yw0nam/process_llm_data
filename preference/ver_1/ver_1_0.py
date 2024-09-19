from inst.basemodel import preprocess
import pandas as pd
import os
from utils import auto_log_process

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    def read_processed_data(self, dicts: dict):
        dicts['generate_novel'] = pd.read_json(os.path.join(self.dataset_path, 'processed/generate_novel_preference.json'))
        dicts['fill_mask'] = pd.read_json(os.path.join(self.dataset_path, 'processed/fill_mask_preference.json'))
        dicts['target_chara_chat'] = pd.read_json(os.path.join(self.dataset_path, 'processed/target_chara_chat_preference.json'))
        dicts['long_context_chat'] = pd.read_json(os.path.join(self.dataset_path, 'processed/long_context_chat_preference.json'))
        return dicts
    def process_datasets(self) -> dict:
        dicts = {}
        dicts = self.read_processed_data(dicts)
        return dicts