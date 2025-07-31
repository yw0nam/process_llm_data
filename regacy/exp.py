# %%
import ast
import datasets
import pandas as pd

# %%
data = datasets.load_from_disk("/tmp/tmppp693hp4/output/demo_dataset")
# %%
data = jload("/data2/datas/LLM/visual_novel/processed/generate_novel.json")
chara_bg_dicts = jload(
    "/data2/datas/LLM/visual_novel/processed/system_dict_slimed.json"
)
system_message_one_chara = """You are {chara}.
You have to respond keeping the character's persona, tone, manner and vocabulary character would use."""
# %%
val_dataset = datasets.load_dataset(
    "json",
    data_files="/data2/datas/LLM/visual_novel/instruction/vn_ver_2.1/train.json",
    split="train",
)
# %%
data = pd.read_json("/data2/datas/LLM/visual_novel/instruction/vn_ver_2.1/train.json")


# %%
def detect_bad_encoding(df):
    bad_rows = []
    for col in df.columns:
        for index, value in df[col].items():
            try:
                str(value).encode("utf-8")
            except UnicodeEncodeError:
                bad_rows.append((index, col, value))
    return bad_rows


bad_data = detect_bad_encoding(data)

if bad_data:
    print("Found problematic characters in the following rows:")
    for row in bad_data:
        print(f"Row {row[0]}, Column '{row[1]}': {repr(row[2])}")
else:
    print("No encoding issues found.")
# %%
process = preprocess("/data2/datas/LLM/visual_novel", True)
# %%
df = process.make_training_sample()
# %%
df["chat_template"] = df["chat_template"].map(
    lambda x: encode_utf8(x) if isinstance(x, list) else x
)
df["source"] = df["source"].map(lambda x: encode_utf8(x) if isinstance(x, list) else x)
# %%
data = df[["chat_template", "source"]]
train, val = train_test_split(
    data, test_size=5000, random_state=1004, stratify=data["source"]
)
# %%
dataset = datasets.DatasetDict(
    {
        "train": datasets.Dataset.from_pandas(train, preserve_index=False),
        "test": datasets.Dataset.from_pandas(val, preserve_index=False),
    }
)
# %%
dataset.save_to_disk("/data2/datas/LLM/visual_novel/instruction/vn_ver_2.1")
# %%
temp = datasets.load_from_disk("/data2/datas/LLM/visual_novel/instruction/vn_ver_2.1")
# %%
temp[""]
# %%
# Create a sample dataset with an integer list column
data = {
    "id": [1, 2, 3, 4, 5],
    "numbers": [
        [10, 20, 30],  # Length 3
        [5, 15],  # Length 2
        [1, 2, 3, 4, 5],  # Length 5
        [100],  # Length 1
        [50, 60, 70, 80],  # Length 4
    ],
}

# Convert to Hugging Face Dataset
temp_1 = datasets.Dataset.from_dict(data)
# %%
temp_1 = temp_1.map(lambda example: {"length": len(example["numbers"])})
# %%
sorted_dataset = temp_1.sort("length", reverse=True)
# %%
sorted_dataset[2]
# %%

data = pd.DataFrame(
    datasets.load_dataset(
        "SicariusSicariiStuff/Bluemoon_Top50MB_Sorted_Fixed", split="train"
    )
)


def Bluemoon_Top50MB_Sorted_Fixed():
    data = pd.DataFrame(
        datasets.load_dataset(
            "SicariusSicariiStuff/Bluemoon_Top50MB_Sorted_Fixed", split="train"
        )
    )

    def to_chat_template(x):
        out_list = []
        for each_line in x:
            if each_line["from"] == "human":
                out_list.append({"content": each_line["value"], "role": "user"})
            elif each_line["from"] == "gpt":
                out_list.append({"content": each_line["value"], "role": "assistant"})
            elif each_line["from"] == "system":
                out_list.append({"content": each_line["value"], "role": "system"})
            else:
                raise ValueError("Got Wrong role name {}".format(each_line["from"]))
        return out_list

    data["chat_template"] = data["conversations"].map(lambda x: to_chat_template(x))
    data["source"] = "SicariusSicariiStuff/Bluemoon_Top50MB_Sorted_Fixed"
    return data


# %%
data = Bluemoon_Top50MB_Sorted_Fixed()
# %%
data["length"] = data["chat_template"].map(lambda x: len(x))
# %%
data["length"].value_counts()
# %%
data = pd.read_json(
    "/data2/datas/LLM/visual_novel/instruction/vn_ver_2.1_updated/train.json"
)
# %%
data.query("source == 'SicariusSicariiStuff/Bluemoon_Top50MB_Sorted_Fixed'")[
    "chat_template"
].iloc[0]
# %%
