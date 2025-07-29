# from basemodel import preprocess
from .ver_1_3 import preprocess
import datasets
import pandas as pd
import os
from tools.utils import auto_log_process, resize_output

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    
    @resize_output(size=5000)
    def Bluemoon_Top50MB_Sorted_Fixed(self):
        data = pd.DataFrame(datasets.load_dataset('SicariusSicariiStuff/Bluemoon_Top50MB_Sorted_Fixed', split='train'))
        def to_chat_template(x):
            out_list = []
            for each_line in x:
                if each_line['from'] == 'human':
                    out_list.append({'content': each_line['value'], 'role': 'user'})
                elif each_line['from'] == 'gpt':
                    out_list.append({'content': each_line['value'], 'role': 'assistant'})
                elif each_line['from'] == 'system':
                    out_list.append({'content': each_line['value'], 'role': 'system'})
                else:
                    raise ValueError('Got Wrong role name {}'.format(each_line['from']))
            return out_list
        data['chat_template'] = data['conversations'].map(lambda x: to_chat_template(x))
        data['source'] = 'SicariusSicariiStuff/Bluemoon_Top50MB_Sorted_Fixed'
        return data
    @resize_output(size=5000)
    def Aratako_Synthetic_JP_EN_Coding_Dataset_801k(self):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Synthetic-JP-EN-Coding-Dataset-801k', split='train'))
        data = data.query("language == 'Japanese'")
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako_Synthetic_JP_EN_Coding_Dataset_801k'
        return data
    
    @resize_output(size=5000)
    def Magpie_Tanuki_8B_97k(self):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Magpie-Tanuki-8B-97k', split='train'))
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako/Magpie-Tanuki-8B-97k'
        return data
    def process_datasets(self) -> dict[pd.DataFrame]:
        dicts = super().process_datasets()
        dicts['Bluemoon_Top50MB_Sorted_Fixed'] = self.Bluemoon_Top50MB_Sorted_Fixed()
        dicts['Aratako_Synthetic_JP_EN_Coding_Dataset_801k'] = self.Aratako_Synthetic_JP_EN_Coding_Dataset_801k()
        dicts['Magpie_Tanuki_8B_97k'] = self.Magpie_Tanuki_8B_97k()
        return dicts