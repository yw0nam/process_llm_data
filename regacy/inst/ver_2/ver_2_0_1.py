# from basemodel import preprocess
from inst.ver_2.ver_2_0 import preprocess
import datasets
import pandas as pd
import os
import ast
from tools.utils import auto_log_process

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    # Overide 
    def HuggingFaceTB_smoltalk(self, sample_per_split=3000, random_state=1004, waifu=True):
        df_ls = []
        data_splits = ['apigen-80k', 'smol-magpie-ultra', 'smol-constraints', 'smol-rewrite', 'smol-summarize', 'everyday-conversations', 'explore-instruct-rewriting', 'longalign', 'metamathqa-50k', 'numina-cot-100k', 'openhermes-100k', 'self-oss-instruct', 'systemchats-30k']
        for data_split in data_splits:
            if data_split == 'apigen-80k':
                data = pd.read_json(os.path.join(self.dataset_path, 'processed/smoltalk_apigen.json'))
                if waifu:
                    data['messages'] = data['waifu_chat_template']
                else:
                    data['messages'] = data['chat_template']
                data['tools'] = data['tools'].map(lambda x: [{'role': 'function', 'content': str(x)}])
                data['messages'] = data['tools'] + data['messages']
                data['length'] = data['messages'].map(lambda x: len(x))
                data = data.query("length != 1")
                data = data[['messages']].sample(50000, random_state=1004)
            else:
                data = pd.DataFrame(datasets.load_dataset('HuggingFaceTB/smoltalk', data_split, split='train'))
                data = data.sample(min(sample_per_split, len(data)), random_state=random_state)
            data['source'] = data_split
            df_ls.append(data) 
        data = pd.concat(df_ls, ignore_index=True)
        data = data.rename({'messages': 'chat_template'}, axis=1)
        return data