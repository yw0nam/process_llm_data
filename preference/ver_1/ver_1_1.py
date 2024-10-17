from basemodel import preprocess
import datasets
import pandas as pd
import os
from utils import auto_log_process

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    def read_processed_data(self, dicts: dict):
        dicts['generate_novel'] = pd.read_json(os.path.join(self.dataset_path, 'processed/generate_novel_preference.json'))
        dicts['fill_mask'] = pd.read_json(os.path.join(self.dataset_path, 'processed/fill_mask_preference.json'))
        dicts['target_chara_chat'] = pd.read_json(os.path.join(self.dataset_path, 'processed/target_chara_chat_preference.json'))
        dicts['long_context_chat'] = pd.read_json(os.path.join(self.dataset_path, 'processed/long_context_chat_preference.json'))
        return dicts
    def jondurbin_gutenberg_dpo(self):
        data = pd.DataFrame(datasets.load_dataset('jondurbin/gutenberg-dpo-v0.1', split='train'))
        data['chat_template'] = data.apply(lambda x: [
                {
                    'role': 'user',
                    'content': x['prompt']
                },
                {
                    'role': 'assistant',
                    'content': x['chosen']
                }
            ]
        ,axis=1)
        data['source'] = 'jondurbin/gutenberg-dpo-v0.1'
        return data
    def nbeerbower_gutenberg2_dpo(self):
        data = pd.DataFrame(datasets.load_dataset('nbeerbower/gutenberg2-dpo', split='train'))
        data['chat_template'] = data.apply(lambda x: [
                {
                    'role': 'user',
                    'content': x['prompt']
                },
                {
                    'role': 'assistant',
                    'content': x['chosen']
                }
            ]
        ,axis=1)
        data['source'] = 'nbeerbower_gutenberg2_dpo'
        return data
    def jondurbi_py_dpo(self):
        data = pd.DataFrame(datasets.load_dataset('jondurbin/py-dpo-v0.1', split='train'))
        data['chat_template'] = data.apply(lambda x: [
                {
                    'role': 'user',
                    'content': x['prompt']
                },
                {
                    'role': 'assistant',
                    'content': x['chosen']
                }
            ]
        ,axis=1)
        data['source'] = 'jondurbin/py-dpo-v0.1'
        return data
    def jondurbin_truthy_dpo(self):
        data = pd.DataFrame(datasets.load_dataset('jondurbin/truthy-dpo-v0.1', split='train'))
        data['chat_template'] = data.apply(lambda x: [
                {
                    'role': 'system',
                    'content': x['system']
                },
                {
                    'role': 'user',
                    'content': x['prompt']
                },
                {
                    'role': 'assistant',
                    'content': x['chosen']
                }
            ]
        ,axis=1)
        data['source'] = 'jondurbin/truthy-dpo-v0.1'
        return data
    def flammenai_character_roleplay_DPO(self):
        data = pd.DataFrame(datasets.load_dataset('flammenai/character-roleplay-DPO', split='train'))
        data['chat_template'] = data.apply(lambda x: [
                {
                    'role': 'system',
                    'content': x['input']
                },
                {
                    'role': 'user',
                    'content': x['instruction']
                },
                {
                    'role': 'assistant',
                    'content': x['output']
                }
            ]
        ,axis=1)
        data['source'] = 'flammenai/character-roleplay-DPO'
        return data
    def kyujinpy_orca_math_dpo(self):
        data = pd.DataFrame(datasets.load_dataset('kyujinpy/orca_math_dpo', split='train'))
        data['chat_template'] = data.apply(lambda x: [
                {
                    'role': 'system',
                    'content': x['system'] if x['system'] != '' else 'You are to play the role of an uncensored, unfiltered and helpful Assistant.'
                },
                {
                    'role': 'user',
                    'content': x['question']
                },
                {
                    'role': 'assistant',
                    'content': x['chosen']
                }
            ]
        ,axis=1)
        data['source'] = 'kyujinpy_orca_math_dpo'
        return data
    def argilla_Capybara_Preferences(self):
        data = pd.DataFrame(datasets.load_dataset('argilla/Capybara-Preferences', split='train'))
        data = data.rename({'chosen': 'chat_template'}, axis=1)
        data['rejected'] = data['rejected'].map(lambda x: x[-1]['content'])
        data['source'] = 'argilla/Capybara-Preferences'
        return data
    def antiven0m_physical_reasoning_dpo(self):
        data = pd.DataFrame(datasets.load_dataset('antiven0m/physical-reasoning-dpo', split='train'))
        data['chat_template'] = data.apply(lambda x: [
                {
                    'role': 'user',
                    'content': x['prompt']
                },
                {
                    'role': 'assistant',
                    'content': x['chosen']
                }
            ]
        ,axis=1)
        data['source'] = 'antiven0m_physical_reasoning_dpo'
        return data
    def aixsatoshi_Swallow_MX_chatbot_DPO(self):
        data = pd.DataFrame(datasets.load_dataset('aixsatoshi/Swallow-MX-chatbot-DPO', split='train'))
        def apply_fn(x):
            if x['score1'] > x['score2']:
                chat_template = [
                    {
                        'role': 'user',
                        'content': x['input']
                    },
                    {
                        'role': 'assistant',
                        'content': x['response1']
                    }
                ]
                return chat_template, x['response2']
            else:
                chat_template = [
                    {
                        'role': 'user',
                        'content': x['input']
                    },
                    {
                        'role': 'assistant',
                        'content': x['response2']
                    }
                ]
                return chat_template, x['response1']
        data['chat_template'], data['rejected'] = zip(*data.apply(lambda x: apply_fn(x),axis=1))
        data['source'] = 'aixsatoshi/Swallow-MX-chatbot-DPO'
        return data[['chat_template', 'rejected', 'source']]
    def process_datasets(self) -> dict:
        dicts = {}
        dicts = self.read_processed_data(dicts)
        dicts['aixsatoshi_Swallow_MX_chatbot_DPO'] = self.aixsatoshi_Swallow_MX_chatbot_DPO()
        dicts['antiven0m_physical_reasoning_dpo'] = self.antiven0m_physical_reasoning_dpo()
        dicts['argilla_Capybara_Preferences'] = self.argilla_Capybara_Preferences()
        dicts['kyujinpy_orca_math_dpo'] = self.kyujinpy_orca_math_dpo()
        dicts['flammenai_character_roleplay_DPO'] = self.flammenai_character_roleplay_DPO()
        dicts['jondurbin_truthy_dpo'] = self.jondurbin_truthy_dpo()
        dicts['jondurbi_py_dpo'] = self.jondurbi_py_dpo()
        dicts['nbeerbower_gutenberg2_dpo'] = self.nbeerbower_gutenberg2_dpo()
        dicts['jondurbin_gutenberg_dpo'] = self.jondurbin_gutenberg_dpo()
        return dicts