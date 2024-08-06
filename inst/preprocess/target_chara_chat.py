# %%
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
import datasets
import random
from tqdm import tqdm
import re
import pandas as pd
from  matplotlib import pyplot as plt
from utils import jdump, jload
# %%
chara_bg_dicts = jload('./../../datas/processed/system_dict_updated.json')
system_message = """This is an RP (roleplay) chat. Our characters could come from visual novels
I'm going to give you an character name, a background.
I want you to respond and answer like characters using the tone, manner and vocabulary characters would use. 
Here is Main Character's background.
"""
# %%
# data = pd.read_csv('./../../data/data.csv')
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
data['name'] = data['name'].fillna('')
# %%
# chara_data = data.query("label == 1") 
main_chara_data = data.query("name in @name_ls")
# %%
main_chara_data['length'] = main_chara_data['text_remove_yomigana'].map(lambda x: len(x))
# main_chara_data.length.hist()
main_chara_data = main_chara_data.query("length > 10")
etc_chara_data = data.query("name not in @name_ls & ~voice.isnull()")
# %%
min_context_window = 5
max_context_window = 10
prev_last_index = 0
out_ls = []
break_flag = 0
for i in tqdm(range(len(main_chara_data))):
    out = []
    index = main_chara_data.index[i]
    context_size = random.randint(min_context_window, max_context_window)
    if data.loc[index]['scene_name'] != data.loc[index-context_size]['scene_name'] or (index - context_size) <= prev_last_index:
        continue
    while True:
        if abs(data.loc[index-context_size]['text_idx'] - data.loc[index-context_size-1]['text_idx']) > 5:
            break
        if data.loc[index-context_size]['name'] == data.loc[index]['name'] or data.loc[index-context_size]['dialog_type'] != 'conversation':
            context_size += 1
        else:
            break
    for j in data.loc[index-context_size:index].index:
        if out == []:
            persona_setup = f"{system_message}{chara_bg_dicts[main_chara_data['name'].iloc[i]]}"
            out.append({
                'role': 'system',
                'content': persona_setup,
            })
            out.append({
                'role': 'user',
                'content': f"{data.loc[j]['name']}:" + data.loc[j]['text_remove_yomigana'],
                'name':data.loc[j]['name']
            })
            continue
        if data.loc[j]['dialog_type'] == 'monologue' and data.loc[j]['name'] == '':  # If, user's Monologue
            out[-1]['content'] = f"{out[-1]['content']}\n{data.loc[j]['text_remove_yomigana']}"
            
        elif out[-1]['name'] == data.loc[j]['name']: # if same character saying continuously
            out[-1]['content'] = f"{out[-1]['content']}\n{data.loc[j]['name']}:{data.loc[j]['text_remove_yomigana']}"
            
        elif out[-1]['name'] != data.loc[j]['name'] and data.loc[index]['name'] != data.loc[j]['name']: # if diff character saying and is not target chara,
            if out[-1]['role'] == 'assistant':
                out.append({
                    'role': 'user',
                    'content': f"{data.loc[j]['name']}: {data.loc[j]['text_remove_yomigana']}",
                    'name':data.loc[j]['name']
                })
            else:
                out[-1]['name'] = data.loc[j]['name']
                out[-1]['content'] = out[-1]['content'] + "\n" +f"{data.loc[j]['name']}: {data.loc[j]['text_remove_yomigana']}"
            
        elif out[-1]['name'] != data.loc[j]['name'] and data.loc[index]['name'] == data.loc[j]['name']: # if diff character saying and is target chara,
            out.append({
                'role': 'assistant',
                'content': f"{data.loc[j]['name']}: {data.loc[j]['text_remove_yomigana']}",
                'name':data.loc[j]['name']
            })
        else:
            break_flag = 1
            break
    if break_flag:
        break
    prev_last_index = index
    if len(out) == 2:
        name = out[1]['name']
        splited = out[1]['content'].split('\n')
        user, assistant = splited[0], "\n".join(splited[1:])
        out[1] = {
            'role': 'user',
            'content': user,
            'name': name
        }
        out.append({
            'role': 'assistant',
            'content': assistant,
            'name': name
        })
    out_ls.append({
        'chat_template': out,
        'character': data.loc[index]['name']
    })
# %%    
df = pd.DataFrame(out_ls)
df['source'] = 'target_chara_chat'
# %%
data= df.apply(lambda x: 
    {   
        "character" : x['character'],
        "chat_template" : x['chat_template'],
        "source" : x['source']
    },
    axis=1
)
# %%
jdump(data.to_list(), './../../datas/processed/target_chara_chat.json')
# %%
