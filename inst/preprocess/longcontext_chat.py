# %%
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
from utils import jload
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import re
import pandas as pd
from utils import jdump
# %%
chara_bg_dicts = jload('./../../datas/processed/system_dict_updated.json')
system_message = """This is an RP (roleplay) chat. Our characters come from visual novels.
I'm going to give you an character's name and background.
I want you to respond and answer like characters using the tone, manner and vocabulary characters would use. 
Here is Main Character's backgrounds.
"""
# %%
data = pd.read_csv('~/Desktop/data/visual_novel/yuzusoft/data.csv')
data = data.loc[3:]
# %%
comp_1 = re.compile("[[][\s0-9ぁ-ゔァ-ヴ々〆〤一-龥ー,\s]*[]]")
comp_2 = re.compile("[[][・][]]")
data['text_remove_yomigana'] = data['text'].map(lambda x: re.sub(comp_1, '', x))
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_2, '', x)) 
# %%ß
comp_3 = re.compile("[『]|[』]")
data['text_remove_yomigana'] = data['text_remove_yomigana'].map(lambda x: re.sub(comp_3, '', x)) 
# data = data[data['text_remove_yomigana'] != "………"]
temp = data['name'].value_counts()[3:]
name_ls = temp[temp > 1500].index.to_list()
# %%
data['name'] = data['name'].replace({'昂晴': 'ユーザー', '暁': 'ユーザー', '将臣': 'ユーザー'})
data['name'] = data['name'].fillna('モノローグ')
# %%
main_chara_data = data.query("name in @name_ls")
main_chara_data['length'] = main_chara_data['text_remove_yomigana'].map(lambda x: len(x))
main_chara_data = main_chara_data.query("length > 10")
etc_chara_data = data.query("name not in @name_ls & ~voice.isnull()")
# %%
min_context_window = 5
max_context_window = 20
prev_last_index = 0
out_ls = []
break_flag = 0
for i in tqdm(range(len(main_chara_data))):
    index = main_chara_data.index[i]
    target_chara = main_chara_data['name'].iloc[i]
    
    context_size = random.randint(min_context_window, max_context_window)
    if data.loc[index]['scene_name'] != data.loc[index-context_size]['scene_name'] or (index - context_size) <= prev_last_index:
        continue
    while True:
        if abs(data.loc[index-context_size]['text_idx'] - data.loc[index-context_size-1]['text_idx']) > 5:
            break
        if context_size == max_context_window:
            break
        if data.loc[index-context_size]['name'] == data.loc[index]['name'] or data.loc[index-context_size]['dialog_type'] != 'conversation':
            context_size += 1
        else:
            break
        
    out = data.loc[index-context_size:index].apply(lambda x:
        f"{x['name']}: {x['text_remove_yomigana']}" if x['name'] != "モノローグ" else
        f"{x['text_remove_yomigana']}"
    ,axis=1).to_list()
    for idx in range(1, len(out)):
        if out[-idx].split(':')[0] != target_chara:
            inst_chat = out[:-idx+1]
            out_chat = out[-idx+1:]
            break
    out_ls.append({
        'chat_template': [
            {
                'role': 'system',
                'content': f"{system_message}{chara_bg_dicts[target_chara]}"
            },
            {
                'role': 'user', 
                "content": "\n".join(inst_chat)
            },
            {
                'role': 'assistant', 
                "content": "\n".join(out_chat)
            }
        ],
        'character': target_chara,
        'scene_name': data.loc[index]['scene_name'],
    })
    prev_last_index = index
# %%
df = pd.DataFrame(out_ls)
df['source'] = 'long_context_chat'
# %%

data = df.apply(lambda x: 
    {   
        "character" : x['character'],
        "chat_template" : x['chat_template'],
        "source" : x['source']
    },
    axis=1
)
# %%
jdump(data.to_list(), './../../datas/processed/long_context_chat.json')
