# %%
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
import pandas as pd
from utils import jload, jdump
# %%
csv = pd.read_json('./../../datas/processed/target_chara_chat.json')
# %%
grouped = csv.groupby(by=['character'])
csv['length'] = csv['chat_template'].map(lambda x: len(x))
# %%
def make_rejected(chat_template, next_chat_assistant, character):
    outs = []
    for length in list(range(3, len(chat_template)+1, 2)):
        out = chat_template[:length]
        try:
            rejected = chat_template[length+1]
        except IndexError:
            rejected = next_chat_assistant
        outs.append({'chat_template': out,'rejected': rejected, 'character':character})
    return outs
# %%
out = []
for name, group in grouped:
    temp_ls = []
    shifted = group['chat_template'].shift(1)
    shifted.iloc[0] = group['chat_template'].iloc[-1]
    shifted = shifted.map(lambda x: x[2])
    group['next_chat_assistant'] = shifted
    chat_with_rejected = group.apply(lambda x: make_rejected(x['chat_template'], x['next_chat_assistant'], x['character']), axis=1)
    chat_with_rejected.map(lambda x: temp_ls.extend(x))
    out += temp_ls
# %%
concated = pd.DataFrame(out)
concated['source'] = 'target_chara_chat'
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
jdump(dataset,'./../../datas/processed/target_chara_chat_preference.json')

# %%
