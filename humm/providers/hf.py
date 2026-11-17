"""HF transformers 로컬 공급자. GPU 한 장에 모델 하나를 올리는 것을 기본으로 한다(device="cuda:0")."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HFProvider:
    supports_logits = True

    def __init__(self, model_id: str, device: str = "cuda:0", dtype=torch.bfloat16, batch_size: int = 8, quant: str = ""):
        """quant(Step 12): "" bf16, "int8"·"nf4" bitsandbytes 가중치 양자화(계산은 dtype)."""
        self.name = model_id
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        qc = {"": None, "int8": BitsAndBytesConfig(load_in_8bit=True),
              "nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)}[quant]
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device, quantization_config=qc).eval()

    def _encode(self, prompts):
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        text = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
        return self.tok(text, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)

    def _batches(self, prompts):
        return (prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size))

    @torch.inference_mode()
    def option_probs(self, prompts, labels):
        ids = [self.tok.encode(l, add_special_tokens=False)[0] for l in labels]
        out = []
        for batch in self._batches(prompts):
            logits = self.model(**self._encode(batch)).logits[:, -1, ids].float()
            out += torch.softmax(logits, -1).tolist()
        return out

    @torch.inference_mode()
    def complete(self, prompts, max_new_tokens=1024):
        out = []
        for batch in self._batches(prompts):
            enc = self._encode(batch)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out
