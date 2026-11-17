# Serving the LM with vLLM

The LM is a standard Hugging Face causal LM, so any OpenAI-compatible server works. hvme only sends completions and parses tool calls itself.

```bash
pip install vllm
vllm serve /path/to/hvme-lm-9b --served-model-name hvme-lm --port 8011 \
  --max-num-seqs 128 --gpu-memory-utilization 0.85
```

Point hvme at the server. `HVME_LM` is still used, but only to load the tokenizer:

```bash
export HVME_LM_URL=http://127.0.0.1:8011/v1
export HVME_LM_NAME=hvme-lm
export HVME_LM=/path/to/hvme-lm-9b
```

```python
from hvme.providers.remote import RemoteProvider
from hvme.lm.sample import generate_tools

lm = RemoteProvider("http://127.0.0.1:8011/v1", "hvme-lm", tokenizer="/path/to/hvme-lm-9b")
print(generate_tools(lm, ["<passage>\n\n<question>"], tools=True, max_new_tokens=2048)[0])
```

In our runs bf16 vLLM gave the same answers as Hugging Face generation on the Gym test set and was much faster. 8-bit and 4-bit bitsandbytes weights were slower than bf16 on our GPUs and lost accuracy, so we do not recommend them.
