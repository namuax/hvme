"""작은 무작위 모델로 Hume-Fast 학습 루프를 확인한다(torch·transformers가 있을 때만)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from humm.fast.judge import FastJudge  # noqa: E402
from humm.fast.router import Routed  # noqa: E402
from humm.fast.train import calibrate, evaluate, predict, train  # noqa: E402

WORDS = ["red", "blue", "green", "cause", "effect", "Yes", "No", "예", "아니오", "A", "B", "C", "D", "E", "1", "2", "3", "4", "5", "is", "the", "item", "color"]


def tiny_judge():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", "User:", "Assistant:"] + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>", padding_side="left")
    tok.chat_template = "{% for m in messages %}User: {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    return FastJudge(Qwen2ForCausalLM(cfg), tok, "tiny", batch_size=8)


def items(n, seed):
    """'red'가 있으면 Yes(70%), 'blue'면 No(70%)인 잡음 섞인 과제."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = str(rng.choice(["red", "blue"]))
        yes = (c == "red") == (rng.random() < 0.7)
        out.append({"id": str(i), "prompt": f"the item color is {c}", "labels": ["Yes", "No"], "target": [float(yes), float(not yes)],
                    "kind": "noul", "task": "toy", "lang": "en", "source": "toy"})
    out += [{"id": f"c{i}", "prompt": f"the cause is {'green' if i % 2 else 'red'}", "labels": ["A", "B"], "target": [float(i % 2), float(1 - i % 2)],
             "kind": "choice", "task": "toy2", "lang": "en", "source": "toy"} for i in range(n // 4)]
    return out


def test_train_calibrate_route():
    judge = tiny_judge()
    tr, dev = items(400, 0), items(200, 1)
    before = evaluate(judge, dev)
    hist = train(judge, tr, epochs=3, lr=3e-3, batch_size=16, grad_accum=1, eval_items=dev, eval_every=50, gradient_checkpointing=False, log=lambda r: None)
    after = evaluate(judge, dev)
    assert hist.loss.iloc[-1] < hist.loss.iloc[0]
    assert after.acc.min() > 0.65 and after.acc.mean() > before.acc.mean()
    temps = calibrate(judge, dev)
    assert set(temps) == {"noul", "choice"} and evaluate(judge, dev).set_index("kind").loc["noul", "ece"] < 0.08
    p = predict(judge, dev[:5])
    assert np.allclose([sum(x) for x in p.probs], 1)
    routed = Routed(judge, judge, threshold=0.99)
    routed.option_probs([d["prompt"] for d in dev[:10]], ["Yes", "No"])
    assert routed.n_fast == 10 and routed.n_slow == 10 and routed.escalation_rate == 1.0


def test_save_load(tmp_path, monkeypatch):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", PreTrainedTokenizerFast.from_pretrained)  # 장난감 토크나이저는 config의 모델 종류(qwen2)를 따르지 않게 한다
    judge = tiny_judge()
    judge.temperature = {"noul": 1.7}
    judge.save(tmp_path / "m")
    j2 = FastJudge.load(str(tmp_path / "m"), device="cpu", dtype=torch.float32)
    assert j2.temperature == {"noul": 1.7} and j2.name == "m"
    p1, p2 = judge.option_probs(["the item color is red"], ["Yes", "No"]), j2.option_probs(["the item color is red"], ["Yes", "No"])
    assert np.allclose(p1, p2, atol=1e-5)


def test_lora_train_and_merge():
    from humm.fast.lora import LoRALinear, add_lora, merge_lora
    judge = tiny_judge()
    n_all = sum(p.numel() for p in judge.model.parameters())
    n_train = add_lora(judge.model, r=4, alpha=8)
    assert 0 < n_train < n_all
    tr, dev = items(400, 0), items(200, 1)
    hist = train(judge, tr, epochs=3, lr=1e-2, batch_size=16, grad_accum=1, eval_items=dev, eval_every=50, gradient_checkpointing=True, log=lambda r: None)
    assert hist.loss.iloc[-1] < hist.loss.iloc[0]
    before = predict(judge, dev[:20])
    merge_lora(judge.model)
    assert not any(isinstance(m, LoRALinear) for m in judge.model.modules())
    after = predict(judge, dev[:20])
    assert np.allclose(np.stack(before.logits), np.stack(after.logits), atol=1e-4)


def test_knowledge_temperature():
    from humm.fast.judge import temp_key
    assert temp_key("You are an expert in x.\nWhich?", ["A", "B"]) == "knowledge:choice" and temp_key("q", ["A", "B"]) == "choice"
    judge = tiny_judge()
    judge.temperature = {"noul": 1.0, "knowledge:noul": 50.0}
    plain, expert = judge.option_probs(["the item color is red", "You are an expert in red"], ["Yes", "No"])
    assert abs(expert[0] - 0.5) < 0.01 and abs(plain[0] - 0.5) > abs(expert[0] - 0.5)


def test_sft_masks_prompt_and_learns():
    from humm.fast.lora import add_lora
    from humm.lm.sft import encode, train_sft
    judge = tiny_judge()
    tok, model = judge.tok, judge.model
    ex = [{"prompt": "the item color is red", "response": "Yes"} if i % 2 else {"prompt": "the item color is blue", "response": "No"} for i in range(64)]
    ids, labels = encode(tok, ex[0], 64)
    assert labels[: len(ids) - 2] == [-100] * (len(ids) - 2) and labels[-1] == tok.eos_token_id  # 풀이(Yes)와 EOS에만 손실
    add_lora(model, r=4, alpha=8)
    hist = train_sft(model, tok, ex, epochs=20, lr=1e-2, batch_size=8, grad_accum=1, max_len=64, log_every=20, log=lambda r: None)
    assert hist.loss.iloc[-1] < hist.loss.iloc[0]
    # 장난감 모델은 고정된 RMSNorm·묶인 출력층 때문에 로짓 폭이 작아 손실이 크게 줄지 않는다. 예측(argmax)이 맞는지로 학습을 확인한다
    for color, ans in [("red", "Yes"), ("blue", "No")]:
        text = tok.apply_chat_template([{"role": "user", "content": f"the item color is {color}"}], tokenize=False, add_generation_prompt=True)
        out = model(**tok(text, return_tensors="pt", add_special_tokens=False)).logits[0, -1]
        assert tok.decode(out.argmax()) == ans


def test_sft_checkpoint_resume_matches_uninterrupted(tmp_path):
    from humm.fast.lora import add_lora
    from humm.lm.sft import train_sft
    ex = [{"prompt": f"the item color is {c}", "response": a} for c, a in [("red", "Yes"), ("blue", "No")] * 12]
    kw = dict(epochs=1, lr=1e-2, batch_size=2, grad_accum=2, max_len=64, log_every=1, log=lambda r: None)

    def run(ckpt=None):
        judge = tiny_judge()
        torch.manual_seed(0)
        add_lora(judge.model, r=4, alpha=8)
        hist = train_sft(judge.model, judge.tok, ex, ckpt=ckpt, ckpt_every=3, **kw)
        return hist, {n: p.detach().clone() for n, p in judge.model.named_parameters()}

    ck = tmp_path / "ck.pt"
    h1, w1 = run(ck)  # 6 step, 3 step에서 저장(마지막 step은 저장하지 않으니 끊긴 학습과 같다)
    assert torch.load(ck, weights_only=False)["step"] == 3
    h2, w2 = run(ck)  # 3 step부터 이어 학습
    _, w3 = run()
    assert h1.loss.tolist() == h2.loss.tolist() and all(torch.equal(w1[n], w2[n]) and torch.equal(w1[n], w3[n]) for n in w1)
