import os
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
from sklearn.model_selection import train_test_split
from utils import jdump
    
def save_split(df, root_path, save_path, type='instruction'):
    if type == 'instruction':
        df = df.apply(lambda x: 
            {   
                "source" : x['source'],
                "chat_template" : x['chat_template']
            },
            axis=1
        )
    elif type == 'preference':
        df = df.apply(lambda x: 
            {   
                "prompt" : x['chat_template'][:-1],
                "chosen" : x['chat_template'][-1]['content'],
                "rejected" : x['rejected'],
                'source': x['source']
            },
            axis=1
        )
    else:
        raise NotImplementedError
    jdump(df.to_list(), os.path.join(root_path, save_path))
    
def process(configs, type):    
    """_summary_

    Args:
        configs (DictConfig): hydra main config
        type (str): For selecting instruction or dpo
    """
    config = configs.instruction if configs.main.process_type == 'instruction' else configs.dpo
    main_config = configs.main

    process_module = instantiate(config)['preprocessor']
    data = process_module.make_training_sample()
    if main_config.sample_size == 'not_selected':
        print("There is no Specified Sample size. Setting sample size as data count")
        dev = data
    elif len(data) > main_config.sample_size:
        dev, _ = train_test_split(data, train_size=main_config.sample_size, random_state=1004, stratify=data['source'])
    elif len(data) < main_config.sample_size:
        print("Specified Sample size is bigger than actual data size\nSetting sample size as data count")
        dev = data
    
    train, val = train_test_split(dev, test_size=main_config.valid_size, random_state=1004, stratify=dev['source'])
    save_split(
        train, 
        configs.dataset_path, 
        os.path.join(type, main_config.version, 'train.json'), 
        type=type
    )
    save_split(
        val, 
        configs.dataset_path, 
        os.path.join(type, main_config.version, 'val.json'),
        type=type
    )

@hydra.main(config_path="./configs/", config_name="config")
def main(configs: DictConfig) -> None:
    
    assert configs.main.version != "not_selected", "You need to specify save path"
    assert configs.main.process_type in ['instruction', "preference"], "Wrong process name. select preference or instruction."
    process(configs, configs.main.process_type)
if __name__ == "__main__":
    main()