from humm.providers.cache import CachedProvider, ReplayProvider


class Counting:
    name, supports_logits, calls = "m", True, 0

    def complete(self, prompts, max_new_tokens=1024):
        self.calls += len(prompts)
        return [p.upper() for p in prompts]

    def option_probs(self, prompts, labels):
        self.calls += len(prompts)
        return [[0.7, 0.3] for _ in prompts]


def test_cache_and_replay(tmp_path):
    inner = Counting()
    c = CachedProvider(inner, tmp_path / "m.json")
    assert c.complete(["a", "b"]) == ["A", "B"] and c.complete(["b", "c"]) == ["B", "C"]
    assert c.option_probs(["q"], ["A", "B"]) == [[0.7, 0.3]] and c.option_probs(["q"], ["A", "B"]) == [[0.7, 0.3]]
    assert inner.calls == 4  # a, b, c, q each called once
    r = ReplayProvider(c.save(), name="m")
    assert r.complete(["c"]) == ["C"] and r.option_probs(["q"], ["A", "B"]) == [[0.7, 0.3]]
