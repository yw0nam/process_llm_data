# from basemodel import preprocess
from inst.ver_2.ver_2_0 import preprocess
import datasets
import pandas as pd
import os
import ast, random
from tools.utils import auto_log_process, resize_output

@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)
    
    @resize_output(size=5000)
    def Aratako_Synthetic_Japanese_Roleplay_NSFW_Claude_3_7s_5_3k_formatted(self):
        data = pd.DataFrame(
            datasets.load_dataset(
                "Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-3.7s-5.3k-formatted",
                split="train",
            )
        )
        data = data.rename({"messages": "chat_template"}, axis=1)
        data["source"] = (
            "Aratako/Synthetic-Japanese-Roleplay-NSFW-Claude-3.7s-5.3k-formatted"
        )
        return data

    def process_datasets(self) -> dict[pd.DataFrame]:
        df_dicts = super().process_datasets()
        df_dicts['Aratako_Synthetic_Japanese_Roleplay_NSFW_Claude_3_7s_5_3k_formatted'] = self.Aratako_Synthetic_Japanese_Roleplay_NSFW_Claude_3_7s_5_3k_formatted()
        return df_dicts
    