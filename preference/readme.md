# Process DPO dataset

## For chat template
```json
{
  "prompt": List[
    {
      "content": str, // Texts about something
      "role": str  // One of 'system', 'user', 'assistant'
    },
    {
      "content": str, // Texts about something
      "role": str  // One of 'system', 'user', 'assistant'
    }
    .
    .
    .
  ],
  "chosen": str,
  "rejected": str
}
```

## For no chat template

```json
{
  "prompt": str,
  "chosen": str,
  "rejected": str,
  "source" : str
}
```