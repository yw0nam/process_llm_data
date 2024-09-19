# %%
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
import pandas as pd
from utils import jload, jdump
# %%
csv = pd.read_json('./../../datas/processed/generate_novel.json')
# %%
grouped = csv.groupby(by=['game_name'])
# %%
out = []
for name, group in grouped:
    shifted = group['output'].shift(1)
    shifted.iloc[0] = group['output'].iloc[-1]
    group['rejected'] = shifted
    out.append(group)
# %%
concated = pd.concat(out)
concated = concated.reset_index(drop=True)
concated = concated[concated['rejected'] != concated['output']]
# %%
novel_generate = concated.apply(lambda x: 
    {   
        "source" : x['source'],
        "instruction": x['instruction'],
        'input': x['input'], 
        'output': x['output'],
        "system": x['system'],
        'rejected': x['rejected']
    },
    axis=1
).to_list()
# %%
jdump(novel_generate,'./../../datas/processed/generate_novel_preference.json')
# %%
