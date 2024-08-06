# %%
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
import pandas as pd
from utils import jload, jdump
# %%
csv = pd.read_json('./../../datas/processed/long_context_chat.json')
# %%
grouped = csv.groupby(by=['character'])
# %%
out = []
for name, group in grouped:
    shifted = group['chat_template'].shift(1)
    shifted.iloc[0] = group['chat_template'].iloc[-1]
    shifted = shifted.map(lambda x: x[-1])
    group['rejected'] = shifted
    out.append(group)
# %%
concated = pd.concat(out)
concated = concated.reset_index(drop=True)
# %%
dataset = concated.apply(lambda x: 
    {   
        "source" : x['source'],
        "character": x['character'],
        'chat_template': x['chat_template'], 
        'rejected': x['rejected']['content']
    },
    axis=1
).to_list()
# %%
jdump(dataset,'./../../datas/processed/long_context_chat_preference.json')
# %%
