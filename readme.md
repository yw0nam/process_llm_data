# Process dataset

Preprocessing datasets for SFT or DPO.

# Quickstart

## Experiment Setting

- Linux
- Python 3.10

a. Create a conda virtual environment and activate it.

```shell
conda create -n LLM_data_processing python=3.10
conda activate LLM_data_processing
```

b. Clone this repository.

```shell
git clone git@bitbucket.org:ibricks-rnd/llm_data_preprocess.git
cd llm_data_preprocess
```

c. Install requirments.

```shell
pip install -r requirements.txt
```

# Process instruction dataset

```shell
python main.py instruction=version_4.1 main.version=ver_4.1 main.process_type=instruction
```

Output data format is as follow

```json
{
  "chat_template": List[
    {
      "content": str,
      "role": str  // One of 'system', 'user', 'assistant'
    }
    ...
  ],
  "source": str,
}
```

In the example:
- `"prompt"` is a list containing dictionaries.
  - `"content"` is a string.
  - `"role"` is a string and can be one of `'system'`, `'user'`, or `'assistant'`.
- `"source"` is a string that indicates where the data comes from."

If you want to make data for dpo, set save_data_for_dpo as true
Note that, this dpo dataset has only prompt and chosen(label). you should build reject yourself. 


# Process DPO dataset

```shell
python main.py dpo=version_2.0_wo_chat main.version=ver_2.0_wo_chat main.process_type=dpo
```