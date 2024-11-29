# from basemodel import preprocess
from .ver_1_2 import preprocess
import datasets
import pandas as pd
import os
from utils import auto_log_process

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    def anthracite_org_stheno_filtered_v1_1(self):
        data = pd.DataFrame(datasets.load_dataset('anthracite-org/stheno-filtered-v1.1', split='train'))
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
        data['source'] = 'anthracite-org/stheno-filtered-v1.1'
        return data

    #OVERRIDE ver 1.2 
    def SkunkworksAI_reasoning_001(self):
        data = pd.DataFrame(datasets.load_dataset('SkunkworksAI/reasoning-0.01', split='train'))
        data['instruction'] = data['instruction']
        data['input'] = ''
        data['source'] = 'SkunkworksAI/reasoning-0.01'
        data['output'] = data['reasoning'] + '\n\n' + data['output']
        return data
    def PJMixers_hieunguyenminh_roleplay_deduped_ShareGPT(self):
        data = pd.DataFrame(datasets.load_dataset('PJMixers/hieunguyenminh_roleplay-deduped-ShareGPT', split='train'))
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
        data['source'] = 'PJMixers/hieunguyenminh_roleplay-deduped-ShareGPT'
        return data

    def Aratako_Rosebleu_1on1_Dialogues_RP(self):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Rosebleu-1on1-Dialogues-RP', 'v2', split='train'))
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako/Rosebleu-1on1-Dialogues-RP_v2'
        return data
    def process_datasets(self) -> dict[pd.DataFrame]:
        dicts = super().process_datasets()
        dicts['anthracite_org_stheno_filtered_v1_1'] = self.anthracite_org_stheno_filtered_v1_1()
        dicts['PJMixers_hieunguyenminh_roleplay_deduped_ShareGPT'] = self.PJMixers_hieunguyenminh_roleplay_deduped_ShareGPT()
        dicts['Aratako_Rosebleu_1on1_Dialogues_RP'] = self.Aratako_Rosebleu_1on1_Dialogues_RP()
        return dicts