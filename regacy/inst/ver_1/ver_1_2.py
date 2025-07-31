from basemodel import preprocess
import datasets
import pandas as pd
import os
from tools.utils import auto_log_process, resize_output

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    def read_processed_data(self, dicts: dict) -> dict[pd.DataFrame]:
        dicts['generate_novel'] = pd.read_json(os.path.join(self.dataset_path, 'processed/generate_novel.json'))
        dicts['fill_mask'] = pd.read_json(os.path.join(self.dataset_path, 'processed/fill_mask.json'))
        dicts['target_chara_chat'] = pd.read_json(os.path.join(self.dataset_path, 'processed/target_chara_chat.json'))
        dicts['long_context_chat'] = pd.read_json(os.path.join(self.dataset_path, 'processed/long_context_chat.json'))
        return dicts
    
    @resize_output(size=5000)
    def Aratako_Synthetic_Japanese_Roleplay_NSFW_Claude(self):
        dataset = datasets.load_dataset('Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-3.5s-15.3k-formatted')
        data_ls = []
        for key in dataset:
            temp_df = pd.DataFrame(dataset[key])
            temp_df = temp_df.rename({'messages': 'chat_template'}, axis=1)
            data_ls.append(temp_df)
        data = pd.concat(data_ls)
        data['source'] = 'Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-3.5s-15.3k-formatted'
        return data
    
    @resize_output(size=10000)
    def Aratako_Synthetic_Japanese_Roleplay_gpt_4o_mini_39_6k_formatted(self):
        dataset = datasets.load_dataset('Aratako/Synthetic-Japanese-Roleplay-gpt-4o-mini-39.6k-formatted')
        data_ls = []
        for key in dataset:
            temp_df = pd.DataFrame(dataset[key])
            temp_df = temp_df.rename({'messages': 'chat_template'}, axis=1)
            data_ls.append(temp_df)
        data = pd.concat(data_ls)
        data['source'] = 'Aratako_Synthetic_Japanese_Roleplay_gpt_4o_mini_39_6k_formatted'
        return data
    
    @resize_output(size=10000)
    def Aratako_Synthetic_JP_EN_Coding_Dataset_Magpie_69k(self):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Synthetic-JP-EN-Coding-Dataset-Magpie-69k', split='train'))
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako/Synthetic-JP-EN-Coding-Dataset-Magpie-69k'
        return data
    
    @resize_output(size=5000)
    def Nopm_Opus_WritingStruct(self):
        data = pd.DataFrame(data = datasets.load_dataset('Nopm/Opus_WritingStruct', split='train'))
        data['chat_template'] = data.rename({'messages': 'chat_template'},axis=1)
        data['source'] = 'Nopm_Opus_WritingStruct'
        return data
    
    @resize_output(size=5000)
    def roleplay4fun_aesir_v1_1(self):
        data = pd.DataFrame(datasets.load_dataset('roleplay4fun/aesir-v1.1', split='train'))
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
        data['source'] = 'roleplay4fun_aesir_v1_1'
        return data


    def process_datasets(self) -> dict[pd.DataFrame]:
        dicts = {}
        dicts = self.read_processed_data(dicts)
        dicts['Aratako_Synthetic_Japanese_Roleplay_NSFW_Claude'] = self.Aratako_Synthetic_Japanese_Roleplay_NSFW_Claude()
        dicts['Aratako_Synthetic_Japanese_Roleplay_gpt_4o_mini_39_6k_formatted'] = self.Aratako_Synthetic_Japanese_Roleplay_gpt_4o_mini_39_6k_formatted()
        dicts['Aratako_Synthetic_JP_EN_Coding_Dataset_Magpie_69k'] = self.Aratako_Synthetic_JP_EN_Coding_Dataset_Magpie_69k()    
        dicts['Nopm_Opus_WritingStruct'] = self.Nopm_Opus_WritingStruct()
        dicts['roleplay4fun_aesir_v1_1'] = self.roleplay4fun_aesir_v1_1()
        return dicts