# %%
import json
import re
import os, sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)
from random import choice
import ast
from transformers import AutoTokenizer
import datasets
import pandas as pd
from utils import jload, jdump
#%%
data = jload('/data2/datas/LLM/visual_novel/processed/generate_novel.json')
chara_bg_dicts = jload('/data2/datas/LLM/visual_novel/processed/system_dict_slimed.json')
system_message_one_chara = """You are {chara} who are expert in composing functions and helpful assistant.
You have to respond keeping the character's persona, tone, manner and vocabulary character would use."""

data = pd.DataFrame(datasets.load_dataset('HuggingFaceTB/smoltalk', 'apigen-80k', split='train'))
#%%
chara_ls = list(chara_bg_dicts.keys())
chara_ls.append(None)
def extract_tools(text):
    try:
        pattern = r'<tools>\[(.*?)\]</tools>'
        match = re.search(pattern, text, re.DOTALL)
        json_part = match.group(0) if match else None
        json_part = re.sub(r'<tools>|</tools>', '', json_part)
        functions = json.loads(json_part)
        tools = []
        for func in functions:
            tools.append({
                "type": "function",
                "function": func
            })
        return tools
    except Exception as e:
        print(f"Error extracting tools: {e}")
        return []
def to_chat_template(messages, system_message_one_chara, chara_bg_dicts, chara_ls, waifu=True):
    
    system = messages[0]['content']
    tools = extract_tools(system)
    system = system.split('You have access to the following tools:')[0]
    if waifu:
        chara = choice(chara_ls)
        if chara != None:
            persona_setup = f"{system_message_one_chara.format_map({'chara': chara})}\n{chara_bg_dicts[chara]}"
            system = system.replace('You are an expert in composing functions.', persona_setup)
        else:
            pass
    out_list = [
        {'role': 'system', 'content': system},
        messages[1],
        {'role': 'tool_calls', 'content': re.sub(r'<tool_call>|</tool_call>', '', messages[2]['content'])},
    ]
    return out_list, tools
# %%
waifu = data['messages'].map(lambda x: to_chat_template(x, system_message_one_chara, chara_bg_dicts, chara_ls))
default = data['messages'].map(lambda x: to_chat_template(x, system_message_one_chara, chara_bg_dicts, chara_ls, waifu=False))
# %%
data['tools'] = waifu.map(lambda x: x[1])
data['waifu_chat_template'] = waifu.map(lambda x: x[0])
data['chat_template'] = default.map(lambda x: x[0])
#%%
data['source'] = 'smoltalk_apigen'
# %%
data= data.apply(lambda x: 
    {   
        "tools" : x['tools'],
        "waifu_chat_template": x['waifu_chat_template'],
        "chat_template" : x['chat_template'],
        "source" : x['source']
    },
    axis=1
)
# %%
jdump(data.to_list(), '//data2/datas/LLM/visual_novel/processed/smoltalk_apigen.json')
# %%
