# from basemodel import preprocess
from .ver_1_2 import preprocess
import datasets
import pandas as pd
import os
from tools.utils import auto_log_process, resize_output

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    
    @resize_output(size=5000)
    def Aratako_Rosebleu_1on1_Dialogues_RP(self):
        data = pd.DataFrame(datasets.load_dataset('Aratako/Rosebleu-1on1-Dialogues-RP', 'v2', split='train'))
        data = data.rename({'messages': 'chat_template'}, axis=1)
        data['source'] = 'Aratako/Rosebleu-1on1-Dialogues-RP_v2'
        return data
    def process_datasets(self) -> dict[pd.DataFrame]:
        dicts = super().process_datasets()
        dicts['Aratako_Rosebleu_1on1_Dialogues_RP'] = self.Aratako_Rosebleu_1on1_Dialogues_RP()
        return dicts