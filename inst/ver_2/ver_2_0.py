# from basemodel import preprocess
from inst.ver_1.ver_1_4 import preprocess
import datasets
import pandas as pd
import os
import ast
from tools.utils import auto_log_process, resize_output


@auto_log_process
class preprocess(preprocess):
    def __init__(self, dataset_path, use_system):
        super().__init__(dataset_path, use_system)

    @resize_output(size=5000)
    def Aratako_Synthetic_JP_EN_Coding_Dataset_801k(self):
        data = pd.DataFrame(
            datasets.load_dataset(
                "Aratako/Synthetic-JP-EN-Coding-Dataset-801k", split="train"
            )
        )
        data = data.query("language == 'Japanese'")
        data = data.rename({"messages": "chat_template"}, axis=1)
        data["source"] = "Aratako_Synthetic_JP_EN_Coding_Dataset_801k"
        return data

    # Overide
    @resize_output(size=5000)
    def Magpie_Tanuki_8B_97k(self):
        data = pd.DataFrame(
            datasets.load_dataset("Aratako/Magpie-Tanuki-8B-97k", split="train")
        )
        data = data.rename({"messages": "chat_template"}, axis=1)
        data["source"] = "Aratako/Magpie-Tanuki-8B-97k"
        return data

    def microsoft_orca_agentinstruct_1M_v1(
        self, sample_per_split=3000, random_state=1004
    ):
        data = datasets.load_dataset("microsoft/orca-agentinstruct-1M-v1")
        df_ls = []

        def to_chat_template(x):
            message = ast.literal_eval(x)
            if message[0]["content"] == "":
                return message[1:]
            else:
                return message

        for key in list(data.column_names.keys()):
            df = pd.DataFrame(data[key])
            df["chat_template"] = df.messages.map(lambda x: to_chat_template(x))
            df = df.sample(min(len(df), sample_per_split), random_state=random_state)
            df_ls.append(df)
        data = pd.concat(df_ls, ignore_index=True)
        data["source"] = "microsoft_orca_agentinstruct_1M_v1"
        return data

    @resize_output(size=5000)
    def Aratako_Synthetic_JP_EN_Translation_Dataset_Magpie_Nemotron(self):
        data = pd.DataFrame(
            datasets.load_dataset(
                "Aratako/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k",
                split="train",
            )
        )
        data = data.rename({"messages": "chat_template"}, axis=1)
        data["source"] = (
            "Aratako/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k"
        )
        return data

    def HuggingFaceTB_smoltalk(
        self, sample_per_split=3000, random_state=1004, waifu=True
    ):
        df_ls = []
        data_splits = [
            "apigen-80k",
            "smol-magpie-ultra",
            "smol-constraints",
            "smol-rewrite",
            "smol-summarize",
            "everyday-conversations",
            "explore-instruct-rewriting",
            "longalign",
            "metamathqa-50k",
            "numina-cot-100k",
            "openhermes-100k",
            "self-oss-instruct",
            "systemchats-30k",
        ]
        for data_split in data_splits:
            if data_split == "apigen-80k":
                data = pd.read_json(
                    os.path.join(self.dataset_path, "processed/smoltalk_apigen.json")
                )
                if waifu:
                    data["messages"] = data["waifu_chat_template"]
                else:
                    data["messages"] = data["chat_template"]
                data["tools"] = data["tools"].map(
                    lambda x: [{"role": "function", "content": str(x)}]
                )
                data["messages"] = data["tools"] + data["messages"]
                data = data[["messages"]].sample(50000, random_state=1004)
            else:
                data = pd.DataFrame(
                    datasets.load_dataset(
                        "HuggingFaceTB/smoltalk", data_split, split="train"
                    )
                )
                data = data.sample(
                    min(sample_per_split, len(data)), random_state=random_state
                )
            df_ls.append(data)
        data = pd.concat(df_ls, ignore_index=True)
        data = data.rename({"messages": "chat_template"}, axis=1)
        data["source"] = "HuggingFaceTB/smoltalk"
        return data

    def process_datasets(self) -> dict[pd.DataFrame]:
        dicts = super().process_datasets()

        dicts["Aratako_Synthetic_JP_EN_Translation_Dataset_Magpie_Nemotron"] = (
            self.Aratako_Synthetic_JP_EN_Translation_Dataset_Magpie_Nemotron()
        )
        dicts["microsoft_orca_agentinstruct_1M_v1"] = (
            self.microsoft_orca_agentinstruct_1M_v1()
        )
        dicts["HuggingFaceTB_smoltalk"] = self.HuggingFaceTB_smoltalk()
        return dicts
