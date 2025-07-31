# Instruction data 

# Expected format while processing

While processing data, keep this format in mind:

```yaml
messages: list[dict] 
    - role: str  # One of ["system", "user", "assistant", "tool"]
    - content: list[dict]
      - type: str 
      - text: str | None
      - image: str | None
      # Note text or image must be present, not both.
    - tool_calls: list[dict] | None
    - name: str | None
source: str
tools: list[dict] | None
images: list[str] | list[PIL.Image] | None
```
## Instruction output format

When save the output of the instruction, all data should be in this format.

Why all data type is string except images?

For the sake of simplicity and compatibility, all data types are converted to string. This allows for easier serialization and deserialization, especially when storing or transmitting data. When using this data, you can convert the strings back to their original types as needed.

```yaml

messages: str
source: str
tools: str | None
images: list[str] | list[PIL.Image] | None
```

## Expected save format

After processing the data, the saving will be huggingface datasets format.
use 'datasets.save_to_disk' to save the data.

# Preference data format

## Expected format while processing

```yaml
messages: list[dict]
    - role: str  # One of ["system", "user", "assistant", "tool"]
    - content: list[dict] 
      - type: str 
      - text: str | None  
      - image: str | None
      # Note text or image must be present, not both.
    - tool_calls: list[dict] | None
    - name: str | None
source: str
tools: list[dict] | None
images: list[str] | list[PIL.Image] | None
rejected: dict
    - role: str  # One of ["assistant", "tool"]
    - content: 
      - type: str 
      - text: str | None
      - image: str | None
      # Note text or image must be present, not both.
    - tool_calls: list[dict] | None
    - name: str | None
```

## Preference output format

When save the output of the preference, all data should be in this format.

```yaml
messages: str
source: str | None
tools: str | None
images: list[str] | list[PIL.Image] | None
rejected: str
```

## Expected save format

After processing the data, the saving will be huggingface datasets format.
use 'datasets.save_to_disk' to save the data.