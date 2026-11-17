"""Query generator. Graphs and SCMs are generated randomly, and all answers are computed with humm.engine.

meta["vocab"]: node -> (Korean name, state1, state0, English name, state1, state0). Lets solution trajectories (humm.lm.solutions) use the same names as the context.

Numeric queries use only SCMs whose effect is at least MARGIN away from the decision threshold (drops ambiguous answers).
Yes/no queries choose the question direction (increase/decrease) so that answers are balanced 50:50.
"""
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np

from ..engine import BinarySCM, d_separated, identify, is_valid_adjustment
from ..engine.identify import causal_nodes
from .graphs import all_dags, mec, random_dag
from .verbalize import Lexicon, context, describe_cpt, describe_graph, josa
from .vocab import TEST_DOMAINS, sample_vars

MARGIN = 0.05


@dataclass
class Item:
    query_type: str
    rung: int  # 1 association, 2 intervention, 3 counterfactual
    kind: str  # noul | choice | score
    lang: str
    domain: str
    context: str
    question: str
    label: int  # noul: 1/0, choice: index of the correct option (0-based), score: correct level (1-based)
    options: list = field(default_factory=list)
    value: float | None = None
    graph: list = field(default_factory=list)
    latent: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        """Convert tuples to lists so values are identical after a JSON round trip."""
        return json.loads(json.dumps(asdict(self), ensure_ascii=False))


def _seed(rng):
    return int(rng.integers(2**31))


def _lex(g, rng, lang, domain):
    return Lexicon(dict(zip(sorted(g.nodes), sample_vars(domain, g.number_of_nodes(), rng))), lang)


def _q(st):
    """The particle '로/으로' attached after a state."""
    return f"'{st}'{josa(st, '로')[len(st):]}"


def _was(st):
    return f"'{st}'{'이었' if josa(st, '이').endswith('이') else '였'}"


# ---------- Numeric queries (Rung 1, 2, 3) ----------
def _effect_setup(rng, mediated=False, need_pair=True):
    for _ in range(50):
        g = random_dag(int(rng.integers(3, 7)), rng, p=0.5, max_parents=2)
        pairs = [(x, y) for x in g for y in sorted(nx.descendants(g, x)) if not mediated or causal_nodes(g, x, y) - {y}]
        if pairs:
            x, y = pairs[rng.integers(len(pairs))]
            return {"g": g, "x": x, "y": y, "med": sorted(causal_nodes(g, x, y) - {y})}
    return None


def _fit(c, rng, fn, threshold=0.0, tries=30):
    for _ in range(tries):
        scm = BinarySCM.random(c["g"], seed=_seed(rng))
        v = fn(scm, c)
        if abs(v - threshold) >= MARGIN:
            return scm, v
    return None, None


def _numeric(qtype, rung, rng, lang, domain, setup, fn, question, threshold=0.0):
    c = setup(rng)
    if c is None:
        return None
    scm, val = _fit(c, rng, fn, threshold)
    if scm is None:
        return None
    lex = _lex(c["g"], rng, lang, domain)
    want = bool(rng.random() < 0.5)
    up = (val > threshold) == want
    meta = {k: v for k, v in c.items() if k != "g"} | {"direction": "up" if up else "down", "cpt": _cpt(scm), "vocab": lex.vocab}
    return Item(qtype, rung, "noul", lang, domain, context(lex, describe_graph(c["g"], lex), describe_cpt(scm, lex)),
                question(lex, c, up), int(want), value=round(val, 6), graph=list(c["g"].edges), meta=meta)


def _cpt(scm):
    """For reproducing the answer: keep the SCM probability tables in a JSON-serializable form."""
    return {v: [float(p) for p in scm.cpt[v]] for v in scm.order}


def _ko_or_en(lex, ko, en):
    return ko if lex.lang == "ko" else en


def gen_marginal(rng, lang, domain):
    def setup(rng):
        g = random_dag(int(rng.integers(3, 7)), rng, p=0.5, max_parents=2)
        return {"g": g, "y": sorted(g.nodes)[-1]}

    q = lambda lex, c, up: _ko_or_en(
        lex, f"전체적으로 {lex.is_(c['y'], 1)}일 확률이 {lex.is_(c['y'], 0)}일 확률보다 {'높은가' if up else '낮은가'}?",
        f"Overall, is it {'more' if up else 'less'} likely that {lex.is_(c['y'], 1)} than that {lex.is_(c['y'], 0)}?")
    return _numeric("marginal", 1, rng, lang, domain, setup, lambda m, c: m.prob(c["y"]), q, threshold=0.5)


def gen_correlation(rng, lang, domain):
    fn = lambda m, c: m.prob(c["y"], given={c["x"]: 1}) - m.prob(c["y"], given={c["x"]: 0})
    q = lambda lex, c, up: _ko_or_en(
        lex, f"{lex.is_(c['x'], 1)}임을 관측하면, {lex.is_(c['x'], 0)}임을 관측할 때보다 {lex.is_(c['y'], 1)}일 확률이 {'높은가' if up else '낮은가'}?",
        f"Is the probability that {lex.is_(c['y'], 1)} {'higher' if up else 'lower'} when we observe that {lex.is_(c['x'], 1)} "
        f"than when we observe that {lex.is_(c['x'], 0)}?")
    return _numeric("correlation", 1, rng, lang, domain, _effect_setup, fn, q)


def _ate_q(lex, c, up, prefix_ko="", prefix_en=""):
    x, y = c["x"], c["y"]
    return _ko_or_en(
        lex, f"{prefix_ko}개입을 통해 {josa(lex.name(x), '을')} 강제로 {_q(lex.state(x, 1))} 정하면, {_q(lex.state(x, 0))} 정할 때보다 "
             f"{lex.is_(y, 1)}일 확률이 {'높아지는가' if up else '낮아지는가'}?",
        f"{prefix_en}If we intervene to set {lex.name(x)} to {lex.state(x, 1)} instead of {lex.state(x, 0)}, "
        f"does the probability that {lex.is_(y, 1)} {'increase' if up else 'decrease'}?")


def gen_ate(rng, lang, domain):
    return _numeric("ate", 2, rng, lang, domain, _effect_setup, lambda m, c: m.ate(c["x"], c["y"]), _ate_q)


def gen_cde(rng, lang, domain):
    def setup(rng):
        c = _effect_setup(rng, mediated=True)
        return c and c | {"fix": {m: int(rng.integers(2)) for m in c["med"]}}

    q = lambda lex, c, up: _ate_q(lex, c, up, f"{lex.conds(c['fix'])}인 상태로 고정하고, ", f"Holding fixed that {lex.conds(c['fix'])}: ")
    return _numeric("cde", 2, rng, lang, domain, setup, lambda m, c: m.cde(c["x"], c["y"], c["fix"]), q)


def gen_ett(rng, lang, domain):
    def q(lex, c, up):
        x, y = c["x"], c["y"]
        return _ko_or_en(
            lex, f"실제로 {lex.is_(x, 1)}인 경우들만 볼 때, 만약 {josa(lex.name(x), '이')} {_was(lex.state(x, 0))}다면 "
                 f"{lex.is_(y, 1)}일 확률은 {'더 낮았겠는가' if up else '더 높았겠는가'}?",
            f"Among cases where {lex.is_(x, 1)}, would the probability that {lex.is_(y, 1)} have been {'lower' if up else 'higher'} "
            f"if {lex.name(x)} had been {lex.state(x, 0)}?")
    return _numeric("ett", 3, rng, lang, domain, _effect_setup, lambda m, c: m.ett(c["x"], c["y"]), q)


def gen_nde(rng, lang, domain):
    def q(lex, c, up):
        x, y, med = c["x"], c["y"], c["med"]
        return _ko_or_en(
            lex, f"{josa(lex.names(med), '을')} 거치는 간접 경로를 제외하고(매개 변수는 {lex.is_(x, 0)}일 때의 값으로 둔다), "
                 f"{josa(lex.name(x), '이')} '{lex.state(x, 0)}'에서 {_q(lex.state(x, 1))} 바뀌면 {lex.is_(y, 1)}일 확률이 {'높아지는가' if up else '낮아지는가'}?",
            f"If we disregard the mediation effect through {lex.names(med)} (keeping the mediators at the values they would take when "
            f"{lex.is_(x, 0)}), would changing {lex.name(x)} from {lex.state(x, 0)} to {lex.state(x, 1)} "
            f"{'increase' if up else 'decrease'} the probability that {lex.is_(y, 1)}?")
    setup = lambda rng: _effect_setup(rng, mediated=True)
    return _numeric("nde", 3, rng, lang, domain, setup, lambda m, c: m.nde(c["x"], c["y"], c["med"]), q)


def gen_nie(rng, lang, domain):
    def q(lex, c, up):
        x, y, med = c["x"], c["y"], c["med"]
        return _ko_or_en(
            lex, f"{lex.is_(x, 0)}인 상태를 유지하면서 {lex.names(med)}만 {lex.is_(x, 1)}일 때의 값으로 바뀐다면, "
                 f"{lex.is_(y, 1)}일 확률이 {'높아지는가' if up else '낮아지는가'}?",
            f"Suppose {lex.is_(x, 0)}, but {lex.names(med)} take the values they would have if {lex.is_(x, 1)}. "
            f"Does the probability that {lex.is_(y, 1)} {'increase' if up else 'decrease'}?")
    setup = lambda rng: _effect_setup(rng, mediated=True)
    return _numeric("nie", 3, rng, lang, domain, setup, lambda m, c: m.nie(c["x"], c["y"], c["med"]), q)


SCORE_EDGES = [-0.3, -0.1, 0.1, 0.3]
SCORE_LEVELS = {
    "ko": ["크게 낮아진다 (−30%p 이하)", "낮아진다 (−30%p ~ −10%p)", "거의 변하지 않는다 (−10%p ~ +10%p)", "높아진다 (+10%p ~ +30%p)", "크게 높아진다 (+30%p 이상)"],
    "en": ["decreases a lot (−30 pp or less)", "decreases (−30 to −10 pp)", "barely changes (−10 to +10 pp)", "increases (+10 to +30 pp)", "increases a lot (+30 pp or more)"],
}


def gen_ate_score(rng, lang, domain):
    c = _effect_setup(rng)
    if c is None:
        return None
    for _ in range(30):
        scm = BinarySCM.random(c["g"], seed=_seed(rng))
        v = scm.ate(c["x"], c["y"])
        if min(abs(v - e) for e in SCORE_EDGES) >= 0.02:
            break
    else:
        return None
    lex = _lex(c["g"], rng, lang, domain)
    x, y = c["x"], c["y"]
    q = _ko_or_en(lex, f"개입을 통해 {josa(lex.name(x), '을')} {_q(lex.state(x, 1))} 정하면, {_q(lex.state(x, 0))} 정할 때와 비교해 {lex.is_(y, 1)}일 확률은 어떻게 변하는가?",
                  f"If we intervene to set {lex.name(x)} to {lex.state(x, 1)} instead of {lex.state(x, 0)}, how does the probability that {lex.is_(y, 1)} change?")
    return Item("ate_score", 2, "score", lang, domain, context(lex, describe_graph(c["g"], lex), describe_cpt(scm, lex)), q,
                int(np.digitize(v, SCORE_EDGES)) + 1, SCORE_LEVELS[lang], round(v, 6), list(c["g"].edges), meta={"x": x, "y": y, "cpt": _cpt(scm), "vocab": lex.vocab})


# ---------- Structural queries (only the graph is given) ----------
def _pick(rng, seq, k):
    return [str(v) for v in rng.choice(seq, k, replace=False)]


def gen_dsep(rng, lang, domain):
    g = random_dag(int(rng.integers(4, 8)), rng, p=0.4, max_parents=3)
    want = bool(rng.random() < 0.5)
    nodes = sorted(g.nodes)
    for _ in range(40):
        a, b = _pick(rng, nodes, 2)
        rest = [v for v in nodes if v not in (a, b)]
        z = _pick(rng, rest, int(rng.integers(0, min(3, len(rest)) + 1)))
        if d_separated(g, a, b, z) == want:
            break
    else:
        return None
    lex = _lex(g, rng, lang, domain)
    pair_ko = f"{josa(lex.name(a), '과')} {josa(lex.name(b), '은')}"
    q = _ko_or_en(lex, f"{lex.names(z)}의 값을 알고 있을 때(조건부), {pair_ko} 서로 독립인가?" if z else f"아무 조건 없이, {pair_ko} 서로 독립인가?",
                  f"Are {lex.name(a)} and {lex.name(b)} independent given {lex.names(z)}?" if z else f"Are {lex.name(a)} and {lex.name(b)} unconditionally independent?")
    return Item("dsep", 1, "noul", lang, domain, context(lex, describe_graph(g, lex)), q, int(want), graph=list(g.edges), meta={"a": a, "b": b, "z": z, "vocab": lex.vocab})


def gen_backadj(rng, lang, domain):
    g = random_dag(int(rng.integers(4, 8)), rng, p=0.45, max_parents=3)
    pairs = [(x, y) for x in g for y in sorted(nx.descendants(g, x))]
    want = bool(rng.random() < 0.5)
    for _ in range(40):
        x, y = pairs[rng.integers(len(pairs))]
        rest = sorted(set(g.nodes) - {x, y})
        z = _pick(rng, rest, int(rng.integers(0, min(2, len(rest)) + 1)))
        if is_valid_adjustment(g, x, y, z) == want:
            break
    else:
        return None
    lex = _lex(g, rng, lang, domain)
    head = _ko_or_en(lex, f"{josa(lex.name(x), '이')} {lex.name(y)}에 미치는 인과 효과를 추정하려 한다. ", f"We want to estimate the causal effect of {lex.name(x)} on {lex.name(y)}. ")
    tail = _ko_or_en(lex, f"{lex.names(z)}만 통제(조정)하면 충분한가?" if z else "아무 변수도 통제하지 않아도 충분한가?",
                     f"Is it sufficient to adjust for {lex.names(z)} only?" if z else "Is it sufficient to adjust for no variable at all?")
    return Item("backadj", 2, "noul", lang, domain, context(lex, describe_graph(g, lex)), head + tail, int(want), graph=list(g.edges), meta={"x": x, "y": y, "z": z, "vocab": lex.vocab})


STRATEGIES = ["backdoor", "frontdoor", "iv", "not_identified", "no_effect"]
STRATEGY_TEXT = {
    "ko": ["백도어 조정으로 식별된다", "프론트도어 조정으로 식별된다", "도구변수로 식별된다", "관측 변수만으로는 식별할 수 없다", "처치가 결과의 원인이 아니므로 인과 효과는 0이다"],
    "en": ["identified by back-door adjustment", "identified by front-door adjustment", "identified by an instrumental variable",
           "not identifiable from the observed variables", "there is no causal effect because the treatment is not a cause of the outcome"],
}
TEMPLATES = {
    "frontdoor": [("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")],
    "iv": [("Z", "X"), ("U", "X"), ("U", "Y"), ("X", "Y")],
    "not_identified": [("U", "X"), ("U", "Y"), ("X", "Y")],
}


def _identify_graph(rng, target):
    if target in TEMPLATES:
        g = nx.DiGraph(TEMPLATES[target])
        for i in range(int(rng.integers(0, 3))):  # append observed nodes to vary the surface form
            w, old = f"W{i}", [v for v in g.nodes if v != "U"]
            g.add_edges_from((w, v) for v in _pick(rng, old, int(rng.integers(1, 3))))
        mapping = dict(zip(rng.permutation(list(g.nodes)), [f"V{i}" for i in range(g.number_of_nodes())]))
        return nx.relabel_nodes(g, mapping), {mapping["U"]}, mapping["X"], mapping["Y"]
    g = random_dag(int(rng.integers(4, 8)), rng, p=0.45, max_parents=3)
    nodes = sorted(g.nodes)
    x, y = _pick(rng, nodes, 2)
    latent = set(_pick(rng, [v for v in nodes if v not in (x, y)], int(rng.integers(0, 2))))
    return g, latent, x, y


def gen_identify(rng, lang, domain):
    target = STRATEGIES[rng.integers(len(STRATEGIES))]
    for _ in range(60):
        g, latent, x, y = _identify_graph(rng, target)
        if identify(g, x, y, latent).strategy == target:
            break
    else:
        return None
    lex = _lex(g, rng, lang, domain)
    order = list(rng.permutation(len(STRATEGIES)))
    q = _ko_or_en(lex, f"{josa(lex.name(x), '이')} {lex.name(y)}에 미치는 인과 효과를 관측 변수만으로 계산할 수 있는가? "
                       "백도어 조정, 프론트도어 조정, 도구변수 순서로 먼저 적용되는 방법을 고르라.",
                  f"Can the causal effect of {lex.name(x)} on {lex.name(y)} be computed from the observed variables? "
                  "Choose the first applicable method in the order: back-door adjustment, front-door adjustment, instrumental variable.")
    return Item("identify", 2, "choice", lang, domain, context(lex, describe_graph(g, lex, latent)), q, order.index(STRATEGIES.index(target)),
                [STRATEGY_TEXT[lang][i] for i in order], graph=list(g.edges), latent=sorted(latent),
                meta={"x": x, "y": y, "strategy": target, "option_order": [STRATEGIES[i] for i in order], "vocab": lex.vocab})


# ---------- Corr2Cause-style: judge a causal hypothesis from statistical relations alone ----------
def _rel(h, g, a, b):
    an = lambda v: nx.ancestors(g, v)
    return {"direct": g.has_edge(a, b),
            "indirect": b in nx.descendants(g, a) and not g.has_edge(a, b),
            "cause": b in nx.descendants(g, a),
            "collider": any(g.has_edge(a, c) and g.has_edge(b, c) for c in g.nodes),
            "confounder": bool((an(a) & an(b)) - {a, b})}[h]


HYP = {
    "ko": {"direct": "{a}{i} {b}의 직접 원인이다.", "indirect": "{a}{n} {b}의 원인이지만 직접 원인은 아니다.", "cause": "{a}{n} {b}의 (직접 또는 간접) 원인이다.",
           "collider": "{a}{w} {b}의 공통 결과(충돌 변수)가 적어도 하나 있다.", "confounder": "{a}{w} {b}의 공통 원인(교란 변수)이 적어도 하나 있다."},
    "en": {"direct": "{a} directly causes {b}.", "indirect": "{a} causes {b}, but not directly.", "cause": "{a} is a (direct or indirect) cause of {b}.",
           "collider": "There exists at least one collider (i.e., common effect) of {a} and {b}.",
           "confounder": "There exists at least one confounder (i.e., common cause) of {a} and {b}."},
}


def gen_corr2cause(rng, lang, domain):
    """Enumerate all hypotheses whose answer is true and those whose answer is false, then pick one from the desired side (answers balanced 50:50)."""
    want = bool(rng.random() < 0.5)
    for _ in range(20):
        n = int(rng.integers(3, 5))
        dags = all_dags(n)
        g = dags[rng.integers(len(dags))]
        members = mec(g)
        nodes = sorted(g.nodes)
        cands = [(h, a, b) for h in HYP["en"] for a in nodes for b in nodes
                 if a != b and all(_rel(h, m, a, b) for m in members) == want]
        if cands:
            h, a, b = cands[rng.integers(len(cands))]
            break
    else:
        return None
    pool = "PQRSTUVWXYZ" if domain in TEST_DOMAINS else "ABCDEFGHJKLMN"
    name = dict(zip(nodes, _pick(rng, list(pool), n)))
    stmts = []
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            pair = f"{josa(name[u], '과')} {josa(name[v], '은')}"
            dep = not d_separated(g, u, v)
            ko = f"{pair} 상관관계가 있다." if dep else f"{pair} 서로 독립이다."
            en = f"{name[u]} correlates with {name[v]}." if dep else f"{name[u]} is independent of {name[v]}."
            stmts.append(ko if lang == "ko" else en)
            rest = [w for w in nodes if w not in (u, v)]
            for k in range(1, len(rest) + 1):
                for s in map(list, combinations(rest, k)):
                    if dep and d_separated(g, u, v, s):
                        cond = ", ".join(name[w] for w in s) if lang == "ko" else " and ".join(name[w] for w in s)
                        stmts.append(f"{josa(cond, '이')} 주어지면 {pair} 독립이다." if lang == "ko" else f"{name[u]} and {name[v]} are independent given {cond}.")
    letters = ", ".join(name[v] for v in nodes)
    intro = (f"변수 {letters} {n}개로 이루어진 닫힌 계를 생각하라. 이 변수들 사이의 통계적 관계는 다음이 전부이다: " if lang == "ko"
             else f"Suppose there is a closed system of {n} variables, {letters}. All the statistical relations among these {n} variables are as follows: ")
    hyp = HYP[lang][h].format(a=name[a], b=name[b], i=josa(name[a], "이")[1:], n=josa(name[a], "은")[1:], w=josa(name[a], "과")[1:])
    q = f"가설: {hyp} 위의 통계적 관계로부터 이 가설이 반드시 참이라고 할 수 있는가?" if lang == "ko" else f"Hypothesis: {hyp} Does the premise necessarily imply the hypothesis?"
    return Item("corr2cause", 1, "noul", lang, domain, intro + " ".join(stmts), q, int(want), graph=list(g.edges),
                meta={"hypothesis": h, "a": a, "b": b, "mec_size": len(members), "names": name})


GENERATORS = {
    "marginal": gen_marginal, "correlation": gen_correlation, "ate": gen_ate, "cde": gen_cde, "ett": gen_ett, "nde": gen_nde, "nie": gen_nie,
    "ate_score": gen_ate_score, "dsep": gen_dsep, "backadj": gen_backadj, "identify": gen_identify, "corr2cause": gen_corr2cause,
}
