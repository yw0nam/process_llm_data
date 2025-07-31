# Process SFT dataset

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