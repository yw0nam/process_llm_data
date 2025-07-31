# %%
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
import random
from tqdm import tqdm
import re
import pandas as pd
from utils import jdump, jload
# %%
chara_bg_dicts = jload('/data2/datas/LLM/visual_novel/processed/system_dict_slimed.json')
system_message_one_chara = """You are {chara}.
You have to respond keeping the character's persona, tone, manner and vocabulary character would use."""
system_message_mulit_chara = """You are {chara}.
When responding, you can speak as any of the characters depending on the context.
You must respond while keeping the character's persona, tone, manner, and vocabulary that each character would use."""
# %%
# data = pd.read_csv('./../../data/data.csv')
data = pd.read_csv('/data2/datas/Speech/vn/visual_novel/data.csv')
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
min_context_window = 10
max_context_window = 20
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
        
    main_chara_ls = list(filter(lambda x: x in name_ls, data.loc[index-context_size:index]['name'].unique()))
    system_message = system_message_mulit_chara
    if len(main_chara_ls) > 2:
        charas = ", ".join(main_chara_ls[:-1]) + ", and " + main_chara_ls[-1]
    elif len(main_chara_ls) == 2:
        charas = " and ".join(main_chara_ls)
    else:
        charas = main_chara_ls[0]
        system_message = system_message_one_chara
        
    chara_bgs = []
    for chara in main_chara_ls:
        chara_bgs.append(chara_bg_dicts[chara])
    chara_bg = "\n".join(chara_bgs)
    
    persona_setup = f"{system_message.format_map({'chara': charas})}\n{chara_bg}"
    for j in data.loc[index-context_size:index].index:
        if out == []:
            out.append({
                'role': 'system',
                'content': persona_setup,
            })
            out.append({
                'role': 'user',
                'content': f"{data.loc[j]['name']}: \"{data.loc[j]['text_remove_yomigana']}\"",
                'name':data.loc[j]['name']
            })
            continue
        if data.loc[j]['dialog_type'] == 'monologue' and data.loc[j]['name'] == '':  # If, user's Monologue
            if out[-1]['role'] == 'assistant':
                out.append({
                    'role': 'user',
                    'content': f"*{data.loc[j]['text_remove_yomigana']}*",
                    'name':data.loc[j]['name']
                })
            else:
                out[-1]['name'] = data.loc[j]['name']
                out[-1]['content'] = out[-1]['content'] + "\n" +f"*{data.loc[j]['text_remove_yomigana']}*"
            
        elif out[-1]['name'] == data.loc[j]['name']: # if same character saying continuously
            out[-1]['content'] = f"{out[-1]['content']}\n{data.loc[j]['name']}: \"{data.loc[j]['text_remove_yomigana']}\""
            
        elif out[-1]['name'] != data.loc[j]['name'] and data.loc[j]['name'] not in main_chara_ls: # if diff character saying and is not target chara,
            if out[-1]['role'] == 'assistant':
                out.append({
                    'role': 'user',
                    'content': f"{data.loc[j]['name']}: \"{data.loc[j]['text_remove_yomigana']}\"",
                    'name':data.loc[j]['name']
                })
            else:
                out[-1]['name'] = data.loc[j]['name']
                out[-1]['content'] = out[-1]['content'] + "\n" +f"{data.loc[j]['name']}: \"{data.loc[j]['text_remove_yomigana']}\""
            
        elif out[-1]['name'] != data.loc[j]['name'] and data.loc[j]['name'] in main_chara_ls: # if diff character saying and is target chara,
            if out[-1]['role'] != 'assistant':
                out.append({
                    'role': 'assistant',
                    'content': f"{data.loc[j]['name']}: \"{data.loc[j]['text_remove_yomigana']}\"",
                    'name':data.loc[j]['name']
                })
            else:
                out[-1]['name'] = data.loc[j]['name']
                out[-1]['content'] = out[-1]['content'] + "\n" +f"{data.loc[j]['name']}: \"{data.loc[j]['text_remove_yomigana']}\""
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
        'character': ','.join(main_chara_ls)
    })
# %%    
df = pd.DataFrame(out_ls)
df['source'] = 'target_chara_chat'
df['num_of_chara'] = df['character'].map(lambda x: len(x.split(',')))
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
jdump(data.to_list(), '/data2/datas/LLM/visual_novel/processed/target_chara_chat.json')
# %%
