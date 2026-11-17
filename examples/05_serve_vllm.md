# Serving the LM with vLLM

The LM is a standard Hugging Face causal LM, so any OpenAI-compatible server works. humm only sends completions and parses tool calls itself.

```bash
pip install vllm
vllm serve /path/to/humm-lm-9b --served-model-name humm-lm --port 8011 \
  --max-num-seqs 128 --gpu-memory-utilization 0.85
```

Point humm at the server. `HUMM_LM` is still used, but only to load the tokenizer:

```bash
export HUMM_LM_URL=http://127.0.0.1:8011/v1
export HUMM_LM_NAME=humm-lm
export HUMM_LM=/path/to/humm-lm-9b
```

```python
from humm.providers.remote import RemoteProvider
from humm.lm.sample import generate_tools

lm = RemoteProvider("http://127.0.0.1:8011/v1", "humm-lm", tokenizer="/path/to/humm-lm-9b")
print(generate_tools(lm, ["<passage>\n\n<question>"], tools=True, max_new_tokens=2048)[0])
```

In our runs bf16 vLLM gave the same answers as Hugging Face generation on the Gym test set and was much faster. 8-bit and 4-bit bitsandbytes weights were slower than bf16 on our GPUs and lost accuracy, so we do not recommend them.
