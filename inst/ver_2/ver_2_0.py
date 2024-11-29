# from basemodel import preprocess
from inst.ver_1.ver_1_4 import preprocess
import datasets
import pandas as pd
import os
import ast
from utils import auto_log_process

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    # Overide 
    def Aratako_Synthetic_JP_EN_Coding_Dataset_801k(self, sample_size=30000, random_state=1004):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Synthetic-JP-EN-Coding-Dataset-801k', split='train'))
        data = data.query("language == 'Japanese'")
        data = data.sample(sample_size, random_state=random_state)
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako_Synthetic_JP_EN_Coding_Dataset_801k'
        return data
    # Overide 
    def Magpie_Tanuki_8B_97k(self, sample_size=30000, random_state=1004):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Magpie-Tanuki-8B-97k', split='train'))
        data = data.sample(sample_size, random_state=random_state)
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako/Magpie-Tanuki-8B-97k'
        return data
    def Team_ACE_ToolACE(self):
        data = pd.DataFrame(datasets.load_dataset('Team-ACE/ToolACE', split='train'))
        def to_chat_template(conversations, system):
            out_list = [{'role': 'system', 'content': system}]
            for each_line in conversations:
                if each_line['from'] == 'assistant':
                    try:
                        if each_line['value'][0] == '[' and each_line['value'][-1] == ']':
                            out_list.append({'content': each_line['value'], 'role': 'tool_calls'})
                        else:
                            out_list.append({'content': each_line['value'], 'role': each_line['from']})
                    except Exception as e:
                        return e
                elif each_line['from'] == 'tool':
                    try:
                        content = ast.literal_eval(each_line['value'])
                    except:
                        try:
                            content = ast.literal_eval(each_line['value'].replace('false', "False").replace('true', 'True').replace('null', 'None'))
                        except Exception as e:
                            return e
                    if len(content) != 1:
                        return 'N'
                    out_list.append({'content': str(content[0]['results']), 'role': each_line['from']})
                else:
                    out_list.append({'content': each_line['value'], 'role': each_line['from']})
            return out_list
        data['chat_template'] = data.apply(lambda x: to_chat_template(x['conversations'], x['system']), axis=1)
        data = data[data["chat_template"].map(lambda x: type(x) == list)]
        data['source'] = 'Team-ACE/ToolACE'
        return data
    def microsoft_orca_agentinstruct_1M_v1(self, sample_per_split=5000, random_state=1004):
        data = datasets.load_dataset('microsoft/orca-agentinstruct-1M-v1')
        df_ls = []
        def to_chat_template(x):
            message = ast.literal_eval(x)
            if message[0]['content'] == '':
                return message[1:]
            else:
                return message
        for key in list(data.column_names.keys()):
            df = pd.DataFrame(data[key])
            df['chat_template'] = df.messages.map(lambda x: to_chat_template(x))
            df = df.sample(min(len(df), sample_per_split), random_state=random_state)
            df_ls.append(df)
        data = pd.concat(df_ls, ignore_index=True)
        data['source'] = 'microsoft_orca_agentinstruct_1M_v1'
        return data
    def Aratako_Synthetic_JP_EN_Translation_Dataset_Magpie_Nemotron(self):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k', split='train'))
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k'
        return data
    def HuggingFaceTB_smoltalk(self, sample_per_split=3000, random_state=1004):
        df_ls = []
        data_splits = ['apigen-80k', 'smol-magpie-ultra', 'smol-constraints', 'smol-rewrite', 'smol-summarize', 'everyday-conversations', 'explore-instruct-rewriting', 'longalign', 'metamathqa-50k', 'numina-cot-100k', 'openhermes-100k', 'self-oss-instruct', 'systemchats-30k']
        for data_split in data_splits:
            data = pd.DataFrame(datasets.load_dataset('HuggingFaceTB/smoltalk', data_split, split='train'))
            if data_split == 'apigen-80k':
                data = data.sample(min(15000, len(data)), random_state=random_state)
            else:
                data = data.sample(min(sample_per_split, len(data)), random_state=random_state)
            df_ls.append(data) 
        data = pd.concat(df_ls, ignore_index=True)
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'HuggingFaceTB/smoltalk'
        return data
    def process_datasets(self) -> dict[pd.DataFrame]:
        dicts = super().process_datasets()
        dicts['Team_ACE_ToolACE'] = self.Team_ACE_ToolACE()
        dicts['Aratako_Synthetic_JP_EN_Translation_Dataset_Magpie_Nemotron'] = self.Aratako_Synthetic_JP_EN_Translation_Dataset_Magpie_Nemotron()
        dicts['microsoft_orca_agentinstruct_1M_v1'] = self.microsoft_orca_agentinstruct_1M_v1()
        dicts['HuggingFaceTB_smoltalk'] = self.HuggingFaceTB_smoltalk()
        return dicts