"""causal_engine tools: the model passes variables, graph, probability statements and a query transcribed from the context, and the engine computes the value.

Calculation errors in the first SFT round came from the model inventing intermediate values that are not in the probability table (sums through mediator nodes).
So the model only decides the graph, the query and the comparison target; this tool does the computation.

Probability statements need not be conditional-probability-table entries (CLadder gives joint, marginal and reverse-conditional probabilities).
The tables of all nodes are treated as parameters and fitted to the statements (least squares); the fit is solved several times from different starting points, and the value counts as identified if they agree.
Every value is defined as 'treated world - baseline world'. If the question asks whether something increases or is more likely, the answer is yes when the value is positive.

causal_discovery tool: Corr2Cause-style structure queries. The model passes the premise's correlation/independence statements as they are and only picks the hypothesis kind.
Enumerating equivalence classes by hand with 5-6 variables went wrong (skeleton exact 35-48%, equivalence-class size exact 2-4%).

causal_graph tool: structure queries given only a graph (d-separation, adjustment-set validity, identification strategy). Checking paths by hand
failed by not removing the first edge of a causal path or by inventing unobserved variables (CLadder tool backadj 0.71, Gym identify 0.85).
"""
import json
import re
from difflib import get_close_matches
from itertools import combinations, product

import networkx as nx
import numpy as np
from scipy.optimize import least_squares

from ..engine import BinarySCM, d_separated, frontdoor_sets, id_identifiable, identify, instruments, is_frontdoor, is_instrument
from ..engine.discovery import HYPOTHESES, equivalence_class, from_relations, holds
from ..engine.dsep import blocked_by, paths_between
from ..engine.identify import forbidden, proper_backdoor_graph

QUERIES = ("marginal", "correlation", "ate", "cde", "ett", "nde", "nie", "pn", "ps")

TOOLS = [{"type": "function", "function": {
    "name": "causal_engine",
    "description": "Compute a causal or probabilistic quantity of a binary causal model. Provide the variables, the causal graph and the probabilities "
                   "stated in the problem; the engine fits the model and returns value = (treatment world) − (baseline world).",
    "parameters": {"type": "object", "properties": {
        "variables": {"type": "object", "description": "Each variable name mapped to its two states, e.g. {\"sales\": [\"high\", \"low\"]}."},
        "graph": {"type": "array", "items": {"type": "string"}, "description": "Direct causal edges, e.g. [\"ad -> sales\"]."},
        "unobserved": {"type": "array", "items": {"type": "string"}, "description": "Unobserved variables (optional)."},
        "probabilities": {"type": "array", "items": {"type": "string"},
                          "description": "Every probability given in the problem, e.g. \"P(sales=high | ad=on, season=peak) = 0.30\" or \"P(ad=on, sales=high) = 0.21\"."},
        "query": {"type": "string", "enum": list(QUERIES), "description":
                  "marginal: P(outcome) − 0.5. correlation: P(outcome | treatment, given) − P(outcome | baseline, given). ate: P(outcome | do(treatment)) − P(outcome | do(baseline)). "
                  "cde: like ate with the variables in `fixed` held by intervention. ett: among units whose actual value is the baseline, "
                  "P(outcome had the treatment been applied) − P(outcome). nde: P(outcome | treatment, mediators as under baseline) − P(outcome | baseline). "
                  "nie: P(outcome | baseline, mediators as under treatment) − P(outcome | baseline). "
                  "pn (probability of necessity): given that the treatment and the outcome actually happened, the probability that the outcome would not have happened under the baseline. "
                  "ps (probability of sufficiency): given that the baseline happened and the outcome did not, the probability that the outcome would have happened under the treatment. "
                  "pn and ps return the `bound` (Tian–Pearl bounds)."},
        "treatment": {"type": "object", "description": "{variable: state} of the compared (or counterfactual) world."},
        "baseline": {"type": "object", "description": "{variable: state} of the reference (or actual) world."},
        "outcome": {"type": "object", "description": "{variable: state} whose probability is compared."},
        "mediators": {"type": "array", "items": {"type": "string"}, "description": "Mediators for nde and nie. Leave empty when the question does not name them: the engine then uses every variable on a directed path from the treatment to the outcome."},
        "bound": {"type": "string", "enum": ["lower", "upper"], "description": "Which bound of pn or ps to return."},
        "fixed": {"type": "object", "description": "{variable: state} held fixed by intervention for cde."},
        "given": {"type": "object", "description": "{variable: state} observed condition for correlation, e.g. a common effect when looking only at the units with that value (optional)."},
    }, "required": ["variables", "graph", "probabilities", "query", "outcome"]},
}}, {"type": "function", "function": {
    "name": "causal_discovery",
    "description": "Decide whether statistical relations among variables imply a causal hypothesis. Provide every correlation and independence "
                   "relation stated in the premise; the engine builds the skeleton, the v-structures and the Markov equivalence class "
                   "and checks the hypothesis in every graph of the class.",
    "parameters": {"type": "object", "properties": {
        "variables": {"type": "array", "items": {"type": "string"}, "description": "All variables of the system, e.g. [\"A\", \"B\", \"C\"]."},
        "relations": {"type": "array", "items": {"type": "string"}, "description":
                      "Every relation in the premise, copied as written, e.g. \"A correlates with B.\", \"A is independent of C.\", "
                      "\"A and C are independent given B and D.\""},
        "hypothesis": {"type": "string", "enum": list(HYPOTHESES), "description":
                       "direct: a directly causes b. via: a causes b through some other variable (a direct edge may also exist). "
                       "cause: a is a direct or indirect cause of b. collider: a and b have a common effect. confounder: a and b have a common cause."},
        "a": {"type": "string"}, "b": {"type": "string"},
    }, "required": ["variables", "relations", "hypothesis", "a", "b"]},
}}, {"type": "function", "function": {
    "name": "causal_graph",
    "description": "Answer a structural question about a causal graph when no probabilities are given: whether a causes b (parent or ancestor), d-separation, "
                   "validity of adjustment, front-door or instrumental-variable sets, or whether a causal effect can be identified. Provide the variables and the direct causal edges; the engine checks every path.",
    "parameters": {"type": "object", "properties": {
        "variables": {"type": "array", "items": {"type": "string"}, "description": "All variables of the graph."},
        "graph": {"type": "array", "items": {"type": "string"}, "description": "Direct causal edges, e.g. [\"ad -> sales\"]."},
        "unobserved": {"type": "array", "items": {"type": "string"}, "description": "Unobserved variables (optional)."},
        "confounded": {"type": "array", "items": {"type": "string"}, "description": "Pairs of declared variables that share an unobserved confounder, e.g. [\"A <-> B\"] (optional)."},
        "query": {"type": "string", "enum": ["path", "dsep", "adjustment", "frontdoor", "instrument", "identify"], "description":
                  "path: is a a parent (direct cause) or an ancestor (direct or indirect cause) of b, following the arrows. "
                  "dsep: are a and b d-separated (independent) given `given`; dependence is not causation, so use path to ask whether a causes b. adjustment: is each set in `sets` a valid adjustment set for the effect of a on b. "
                  "frontdoor: does each set in `sets` satisfy the front-door criterion for the effect of a on b. "
                  "instrument: is each set in `sets` a set of instrumental variables for the effect of a on b (each a direct cause of a, connected to b only through a). "
                  "identify: can the effect of a on b be identified from the observed variables (back-door, then front-door, then instrumental variable; with `confounded`, the complete ID algorithm decides)."},
        "a": {"type": "string", "description": "The first variable (the treatment for adjustment and identify)."},
        "b": {"type": "string", "description": "The second variable (the outcome for adjustment and identify)."},
        "given": {"type": "array", "items": {"type": "string"}, "description": "Conditioning variables for dsep."},
        "sets": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "description": "Sets to check for adjustment, frontdoor or instrument, e.g. [[], [\"season\"]]."},
    }, "required": ["variables", "graph", "query", "a", "b"]},
}}]
GRAPH_QUERIES = ("path", "dsep", "adjustment", "frontdoor", "instrument", "identify")

STMT = re.compile(r"^\s*P\s*\((.+?)\)\s*=\s*([0-9.]+)\s*(%?)\s*$")


class ToolError(ValueError):
    pass


def _assign(text: str, states: dict) -> dict:
    """'sales=high, ad=on' -> {'sales': 1, 'ad': 1}. The first state in the list is 1.
    Also reads CLadder-style 'not kwox and swoq': a bare name is the first state, 'not name' is the second. 'gender=not male' is a different state.
    State names are compared ignoring case, whitespace, hyphens and apostrophes ('non-smoker' = 'nonsmoker')."""
    out = {}
    for part in filter(None, (s.strip() for s in re.split(r",|\s+and\s+", text))):
        name, eq, st = (s.strip() for s in part.rpartition("="))
        if not eq:
            neg = part.startswith("not ") and part[4:].strip() in states
            name = part[4:].strip() if neg else part
            st = states[name][int(neg)] if name in states else ""
        if name not in states:
            raise ToolError(f"cannot read '{part}': " + (_undeclared([name], states) if eq else "use name=state with a declared variable and one of its two states"))
        keys = [_norm(s) for s in states[name]]
        if st.lower().startswith("not ") and _norm(st[4:]) in keys:  # 'gender=not male' -> whichever of the two states is not male
            st = states[name][1 - keys.index(_norm(st[4:]))]
        if _norm(st) not in keys:
            raise ToolError(f"cannot read '{part}': {name} was declared with the states {' / '.join(states[name])}; use one of them")
        out[name] = 1 - keys.index(_norm(st))
    return out


def _undeclared(names, declared) -> str:
    """Point out names that were not declared. When a meaningless variable name is mistyped while transcribing (fjfq -> fjq), the model needs to know which name is wrong in order to fix it."""
    return (f"{', '.join(names)} {'is not a declared variable' if len(names) == 1 else 'are not declared variables'} (declared: {', '.join(declared)}); "
            "spell each name exactly as in the question, the same way in variables, graph and probabilities")


def _norm(state: str) -> str:
    return re.sub(r"[\s\-'’_]", "", state.lower())


def parse_statement(s: str, states: dict) -> tuple[dict, dict, float]:
    """'P(Y=a | X=b) = 0.30' → (target, given, p)."""
    m = STMT.match(s)
    if not m:
        raise ToolError(f"cannot read probability '{s}': expected P(A=a | B=b) = 0.30")
    body, num, pct = m.groups()
    target, _, given = body.partition("|")
    p = float(num) / (100 if pct else 1)
    if not 0 <= p <= 1:
        raise ToolError(f"probability out of range in '{s}'")
    return _assign(target, states), _assign(given, states), p


class Model:
    """Binary probability-table model over the declared variables and graph. Probabilities are computed by enumerating all assignments (2^n)."""

    def __init__(self, states: dict, edges: list[tuple[str, str]]):
        self.g = nx.DiGraph(edges)
        self.g.add_nodes_from(states)
        self.lone = [v for v in states if edges and not self.g.degree(v)]  # declared variables that have no graph edge
        if not nx.is_directed_acyclic_graph(self.g):
            raise ToolError("the graph has a cycle")
        self.nodes = list(nx.topological_sort(self.g))
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.parents = {v: sorted(self.g.predecessors(v)) for v in self.nodes}
        self.grid = np.array(list(product((0, 1), repeat=len(self.nodes))))  # (2^n, n)
        self.sizes = [2 ** len(self.parents[v]) for v in self.nodes]
        self.rows = [sum(self.grid[:, self.idx[p]] << i for i, p in enumerate(self.parents[v])) if self.parents[v] else np.zeros(len(self.grid), int)
                     for v in self.nodes]

    def split(self, theta) -> dict:
        out, k = {}, 0
        for v, n in zip(self.nodes, self.sizes):
            out[v] = theta[k:k + n]
            k += n
        return out

    def joint(self, cpt: dict) -> np.ndarray:
        w = np.ones(len(self.grid))
        for v, r in zip(self.nodes, self.rows):
            p = cpt[v][r]
            w = w * np.where(self.grid[:, self.idx[v]] == 1, p, 1 - p)
        return w

    def mask(self, assign: dict) -> np.ndarray:
        m = np.ones(len(self.grid), bool)
        for v, x in assign.items():
            m &= self.grid[:, self.idx[v]] == x
        return m

    def cond(self, w, target: dict, given: dict) -> float:
        den = w[self.mask(given)].sum()
        return float(w[self.mask(target | given)].sum() / max(den, 1e-12))

    def cpt_entry(self, target: dict, given: dict):
        """If the statement is exactly a probability-table entry, return (node, row index, target value)."""
        if len(target) != 1:
            return None
        (v, x), = target.items()
        if set(given) != set(self.parents[v]):
            return None
        return v, sum(given[p] << i for i, p in enumerate(self.parents[v])), x

    def fit(self, stmts: list, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0.2, 0.8, sum(self.sizes))
        fixed = np.zeros(len(theta), bool)
        offsets = dict(zip(self.nodes, np.cumsum([0] + self.sizes[:-1])))
        rest = []
        for t, gv, p in stmts:
            e = self.cpt_entry(t, gv)
            if e:
                v, row, x = e
                theta[offsets[v] + row], fixed[offsets[v] + row] = (p if x else 1 - p), True
            else:
                rest.append((t, gv, p))
        free = ~fixed

        def resid(z):
            th = theta.copy()
            th[free] = 1 / (1 + np.exp(-z))
            w = self.joint(self.split(th))
            return np.array([self.cond(w, t, gv) - p for t, gv, p in rest] or [0.0])

        if rest and free.any():
            z0 = np.log(theta[free] / (1 - theta[free]))
            z = least_squares(resid, z0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000).x
            theta[free] = 1 / (1 + np.exp(-z))
        th = np.clip(theta, 1e-9, 1 - 1e-9)
        cpt = self.split(th)
        w = self.joint(cpt)
        err = max([abs(self.cond(w, t, gv) - p) for t, gv, p in stmts] or [0.0])
        if err > 2e-3:  # separately declared variables are flagged only on a mismatch (raising an error just for an edge-less variable broke calls that were otherwise correct)
            hint = f": {self.lone[0]} is declared but is in no edge of the graph; name each variable once, the same way in variables, graph, probabilities, treatment and outcome" if self.lone else ""
            raise ToolError(f"the probabilities are inconsistent with the graph (max error {err:.3f}){hint}")
        return cpt


def evaluate(scm: BinarySCM, q: dict) -> tuple[float, list[tuple[str, float]]]:
    """(value, [(term name, value)])."""
    query, (y, o) = q["query"], next(iter(q["outcome"].items()))
    eq = lambda vals: (vals[y] == o).astype(float)
    if query == "marginal":
        p = scm._mean(eq(scm.world()))
        return p - 0.5, [("P({o})", p)]
    (x, t), (_, b) = next(iter(q["treatment"].items())), next(iter(q["baseline"].items()))
    f = scm.world()
    if query == "correlation":
        gv = np.prod([f[k] == v for k, v in q.get("given", {}).items()], axis=0).astype(bool) if q.get("given") else True
        a, c = scm._mean(eq(f), (f[x] == t) & gv), scm._mean(eq(f), (f[x] == b) & gv)
        return a - c, [("P({o} | {t}{g})", a), ("P({o} | {b}{g})", c)]
    if query in ("ate", "cde"):
        fix = q.get("fixed", {}) if query == "cde" else {}
        a, c = scm._mean(eq(scm.world({x: t, **fix}))), scm._mean(eq(scm.world({x: b, **fix})))
        return a - c, [("P({o} | do({t}{f}))", a), ("P({o} | do({b}{f}))", c)]
    if query == "ett":
        a, c = scm._mean(eq(scm.world({x: t})), f[x] == b), scm._mean(eq(f), f[x] == b)
        return a - c, [("P({o} had {t} | actually {b})", a), ("P({o} | {b})", c)]
    if query in ("pn", "ps"):  # Tian-Pearl upper/lower bounds. y = outcome state, x = treatment, x' = baseline
        py, pxy, pnn = scm._mean(eq(f)), scm._mean(eq(f) * (f[x] == t)), scm._mean((1 - eq(f)) * (f[x] == b))
        dyt, dyb = scm._mean(eq(scm.world({x: t}))), scm._mean(eq(scm.world({x: b})))
        lower = q["bound"] == "lower"
        if query == "pn":
            terms = [("P({o})", py), ("P({o} | do({b}))", dyb), ("P({t}, {o})", pxy)] if lower else \
                [("P(not {o} | do({b}))", 1 - dyb), ("P({b}, not {o})", pnn), ("P({t}, {o})", pxy)]
        else:
            terms = [("P({o} | do({t}))", dyt), ("P({o})", py), ("P({b}, not {o})", pnn)] if lower else \
                [("P({o} | do({t}))", dyt), ("P({t}, {o})", pxy), ("P({b}, not {o})", pnn)]
        (_, a), (_, c), (_, d) = terms
        return (max(0.0, (a - c) / d) if lower else min(1.0, (a - c) / d)), terms
    med = q["mediators"]
    wt, wb = scm.world({x: t}), scm.world({x: b})
    cross = scm.world({x: t, **{m: wb[m] for m in med}}) if query == "nde" else scm.world({x: b, **{m: wt[m] for m in med}})
    a, c = scm._mean(eq(cross)), scm._mean(eq(wb))
    name = "P({o} | do({t}), {m} as under {b})" if query == "nde" else "P({o} | do({b}), {m} as under {t})"
    return a - c, [(name, a), ("P({o} | do({b}))", c)]


def _redeclare(states: dict, args: dict) -> tuple[dict, dict]:
    """If the declared states differ from those used in the call, read the used states. On CauGym the model often invented states in the declaration (high / normal) but wrote the probabilities as in the passage (low),
    got the same error three times and ended. Only rewrite when two used states exist and one of them is undeclared, or when one used state is undeclared and one declared state is unused."""
    text = json.dumps({k: args.get(k) for k in ("probabilities", "treatment", "baseline", "outcome", "fixed", "given")}, ensure_ascii=False).replace('": "', "=")
    used = {}
    for part in re.split(r'[,|()"\[\]{}]|\s+and\s+', text):
        name, eq, st = (x.strip() for x in part.rpartition("="))
        if eq and name in states and st and not st.lower().startswith("not ") and _norm(st) not in map(_norm, used.setdefault(name, [])):
            used[name].append(st)
    out, changed = dict(states), {}
    for v, st in states.items():
        used_v = used.get(v, [])
        keys = [_norm(x) for x in st]
        new = [u for u in used_v if _norm(u) not in keys]
        if not new or len(used_v) > 2:
            continue
        if len(used_v) == 2:
            kept = [u for u in used_v if _norm(u) in keys]
            fresh = [kept[0], new[0]] if kept and keys.index(_norm(kept[0])) == 0 else [new[0], kept[0]] if kept else used_v
        else:
            unused = [x for x in st if _norm(x) not in map(_norm, used_v)]
            if len(unused) != 1:
                continue
            fresh = [new[0] if x == unused[0] else x for x in st]
        out[v], changed[v] = fresh, (st, fresh)
    return out, changed


def _implied_edges(g: nx.DiGraph, stmts: list) -> list[tuple[str, str]]:
    """Edges that add, as parents, the conditioning variables of probability statements that are not already parents of the target variable (those creating a cycle are skipped)."""
    g, added = g.copy(), []
    for tg, gv, _ in stmts:
        for v in tg if len(tg) == 1 else ():
            for u in gv:
                if u != v and not g.has_edge(u, v) and not nx.has_path(g, v, u):
                    g.add_edge(u, v)
                    added.append((u, v))
    return added


def run(args: dict, restarts: int = 3, repair: bool = True) -> tuple[float, str]:
    """Execute one tool call. Returns (value, result sentence to give back to the model). Invalid input raises ToolError.
    repair: read states that differ from the declaration as the used states (_redeclare), and if the probabilities do not fit the graph, try adding the conditioning variables as parents (_implied_edges). Turned off when building recovery trajectories."""
    states = {k: [str(s) for s in v] for k, v in args["variables"].items()}
    if any(len(v) != 2 or v[0] == v[1] for v in states.values()):
        raise ToolError("every variable needs exactly two different states")
    states, redeclared = _redeclare(states, args) if repair else (states, {})
    edges = []
    for e in args["graph"]:
        u, arrow, v = (s.strip() for s in str(e).partition("->"))
        if not arrow:
            raise ToolError(f"cannot read edge '{e}': use 'A -> B' with declared variables")
        if {u, v} - set(states):
            raise ToolError(f"cannot read edge '{e}': " + _undeclared([x for x in (u, v) if x not in states], states))
        edges.append((u, v))
    query = args["query"]
    if query not in QUERIES:
        raise ToolError(f"unknown query '{query}'")
    q = {"query": query, **{k: _assign(", ".join(f"{a}={b}" for a, b in args.get(k, {}).items()), states) for k in ("treatment", "baseline", "outcome", "fixed", "given")}}
    q["mediators"] = [str(m) for m in args.get("mediators", [])]
    q["bound"] = args.get("bound")
    if len(q["outcome"]) != 1 or (query != "marginal" and (len(q["treatment"]) != 1 or q["treatment"].keys() != q["baseline"].keys())):
        raise ToolError("give one outcome, and for this query one treatment and a baseline on the same variable")
    if set(q["fixed"]) & set(q["treatment"]) or (query != "cde" and q["fixed"]):
        raise ToolError("`fixed` is only for cde and must not contain the treatment variable")
    if q["given"] and (query != "correlation" or set(q["given"]) & {*q["treatment"], *q["outcome"]}):
        raise ToolError("`given` is only for correlation and must not contain the treatment or the outcome")
    if query in ("pn", "ps") and q["bound"] not in ("lower", "upper"):
        raise ToolError("pn and ps need bound: lower or upper")
    if {*q["treatment"], *q["outcome"]} & set(q["mediators"]):  # putting the outcome in the mediator set gave a value of 0
        raise ToolError("mediators are the variables between the treatment and the outcome: leave the treatment and the outcome out of them")
    model = Model(states, edges)
    if query in ("nde", "nie") and not q["mediators"]:  # an empty mediator set means all variables on treatment -> outcome paths (if none, nde = ate, nie = 0)
        (x, _), (y, _) = next(iter(q["treatment"].items())), next(iter(q["outcome"].items()))
        q["mediators"] = sorted(nx.descendants(model.g, x) & nx.ancestors(model.g, y), key=model.nodes.index)
    stmts = [parse_statement(s, states) for s in args["probabilities"]]
    seen = {}
    for s, (tg, gv, p) in zip(args["probabilities"], stmts):  # a mistake of writing the same condition twice points to both statements instead of reporting a mismatch
        key = (tuple(sorted(tg.items())), tuple(sorted(gv.items())))
        if key in seen and abs(seen[key][1] - p) > 1e-9:
            raise ToolError(f"'{seen[key][0]}' and '{s}' give the same event under the same condition: each condition appears once, so one of them has a wrong state")
        seen[key] = (s, p)
    added = []
    try:
        model.fit(stmts, seed=0)
    except ToolError as e:  # a probability conditioned on a non-parent variable that varies with that variable is a dependence missing from the graph; if adding that edge makes it fit, compute with that graph
        added = _implied_edges(model.g, stmts) if repair else []
        if not str(e).startswith("the probabilities are inconsistent") or not added:
            raise
        model = Model(states, edges + added)
        model.fit(stmts, seed=0)
    vals, terms = [], None
    for r in range(restarts * 4):  # if values disagree, run more fits to see whether at least the sign is determined
        if r == restarts and max(vals) - min(vals) <= 2e-3:
            break
        cpt = model.fit(stmts, seed=r)
        scm = BinarySCM(model.g, {v: np.asarray(p) for v, p in cpt.items()}, max_cells=2_000_000)
        v, terms_r = evaluate(scm, q)
        vals.append(v)
        terms = terms or terms_r
    lo, hi = min(vals), max(vals)
    if hi - lo > 2e-3 and (lo * hi <= 0 or query in ("pn", "ps")):
        raise ToolError(f"not identifiable from the given probabilities (value ranges from {lo:+.4f} to {hi:+.4f})")
    val = float(np.mean(vals)) + 0.0  # −0.0 → 0.0
    raw = {k: ", ".join(f"{a}={b}" for a, b in args.get(key, {}).items()) for k, key in (("o", "outcome"), ("t", "treatment"), ("b", "baseline"))}
    raw |= {"g": "".join(f", {a}={b}" for a, b in args.get("given", {}).items()), "m": ", ".join(q["mediators"]) or "no mediators", "f": "".join(f", {a}={b}" for a, b in args.get("fixed", {}).items()) if query == "cde" else ""}
    names = [n.format(**raw) for n, _ in terms]
    diff = f"{names[0]} − {names[1]}" if len(terms) == 2 else f"{names[0]} − 0.5"
    if query in ("pn", "ps"):
        diff = f"{'max{0' if q['bound'] == 'lower' else 'min{1'}, ({names[0]} − {names[1]}) / {names[2]}}}"
    lines = [f"{n} = {p:.4f}" for n, (_, p) in zip(names, terms)] + [f"value = {diff} = {val:{'.4f' if query in ('pn', 'ps') else '+.4f'}}"]
    if not args["probabilities"]:  # value determined without probabilities (0 because no path exists): term values of an arbitrary fit are not shown
        lines = [f"value = {diff} = {val:+.4f} (determined by the graph alone, without probabilities)"]
    if query in ("nde", "nie") and not args.get("mediators"):
        lines.insert(0, f"mediators (variables on a directed path from the treatment to the outcome): {raw['m']}")
    if query == "ett":  # the ETT value in CaLM and CauGym uses the textbook definition (actual - counterfactual). value (counterfactual - actual) gives the sign for 'would it have been more likely'
        lines.append(f"ETT = E[outcome actual − outcome had {raw['t']} | actually {raw['b']}] = {names[1]} − {names[0]} = {0.0 - val:+.4f} "
                     "(the effect of treatment on the treated, actual minus counterfactual)")
    if hi - lo > 2e-3:  # not point-identified, but the sign is the same across all fits
        lines = [f"value = {diff} ranges from {lo:+.4f} to {hi:+.4f}: not point-identified, but the sign is {'positive' if lo > 0 else 'negative'} in every fit"]
    lines += [f"(note: {v} was declared with the states {' / '.join(a)}, but the call uses {' / '.join(b)}; the engine read its states as {' / '.join(b)})"
              for v, (a, b) in redeclared.items()]
    if added:
        lines.append(f"(note: the probabilities are inconsistent with the graph as given, because some conditional probabilities change with a variable "
                     f"that is not a parent; the engine added {', '.join(f'{u} -> {v}' for u, v in added)} and computed the value on that graph)")
    return val, "\n".join(lines)


def _names_in(text: str, names: list[str]) -> list[str]:
    pat = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return re.findall(rf"(?<![A-Za-z0-9_])(?:{pat})(?![A-Za-z0-9_])", text)


KO_GIVEN = re.compile(r"주어지면|주어졌을 ?때|주어질 ?때|주어진 경우|주어진 상태에서|조건으로|조건부로|통제하면|통제했을 ?때|고정하면|고정했을 ?때"
                      r"|주었을 ?때|준 ?경우|알 ?때|알려졌을 ?때|(?:조건 ?)?하에서|일 ?때")


def parse_relation(s: str, names: list[str]) -> tuple[frozenset, frozenset | None]:
    """'A and B are independent given C and D.' → ({A, B}, {C, D}), 'A correlates with B.' → ({A, B}, None). Also reads Korean Gym sentences."""
    indep = "independent" in s or "독립" in s
    if not indep and "correlat" not in s and "상관" not in s:
        raise ToolError(f"cannot read relation '{s}': expected 'X correlates with Y.' or 'X and Y are independent given Z.'")
    pat = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    m = None
    if indep and re.search("[가-힣]", s):
        m = (re.search(rf"(?<![A-Za-z0-9_])({pat}) ?[와과] ?({pat}) ?[은는이가]? ?(?:서로 ?)?(?:조건부 ?)?독립", s)
             or re.search(rf"(?<![A-Za-z0-9_])({pat}) ?[와과] ?({pat}) ?[은는]", s))
    ko = KO_GIVEN.search(s)  # In Korean the condition comes first: 'C 가 주어졌을 때 A 와 B 는 독립이다.'
    if m:  # Korean independence sentence 'X 와 Y 는 독립': every other name, before or after it, is conditioned on, whatever the wording
        pair = [m.group(1), m.group(2)]
        given = [n for n in _names_in(s[:m.start()] + " " + s[m.end():], names) if n not in pair]
    else:
        cond, main = (s[:ko.start()], s[ko.end():]) if ko else (s.partition("given ")[2], s.partition("given ")[0])
        pair, given = _names_in(main, names), _names_in(cond, names)
    if len(set(pair)) != 2 or set(given) & set(pair) or (given and not indep):
        raise ToolError(f"cannot read relation '{s}': it must name two declared variables (and the conditioning variables after 'given')")
    return frozenset(pair), frozenset(given) if indep else None


def run_discovery(args: dict) -> tuple[bool, str]:
    """Execute one structure-query tool call. Returns (whether it is implied, result sentence to give back to the model)."""
    names = [str(v) for v in args["variables"]]
    h, a, b = args["hypothesis"], str(args["a"]), str(args["b"])
    if h not in HYPOTHESES or a not in names or b not in names or a == b:
        raise ToolError(f"hypothesis must be one of {', '.join(HYPOTHESES)} with two different declared variables a and b")
    sep, dep = {}, set()
    for s in args["relations"]:
        pair, given = parse_relation(str(s), names)
        if given is None:
            dep.add(pair)
        else:
            sep.setdefault(pair, []).append(given)
    missing = [tuple(sorted(p)) for p in map(frozenset, combinations(names, 2)) if p not in dep and frozenset() not in sep.get(p, [])]
    if missing or dep & {p for p, ss in sep.items() if frozenset() in ss}:
        raise ToolError(f"every pair needs its marginal relation once: missing {', '.join(f'{u}–{v}' for u, v in missing)}" if missing
                        else "a pair is stated both as correlated and as independent")
    adj, vs = from_relations(names, sep)
    members = equivalence_class(names, adj, vs)
    if not members or any(not d_separated(members[0], *sorted(p), s) for p, ss in sep.items() for s in ss) \
            or any(d_separated(members[0], *sorted(p)) for p in dep):
        raise ToolError("no DAG is consistent with these relations: copy every relation of the premise exactly")
    k = sum(holds(h, g, a, b) for g in members)
    lines = [f"skeleton: {', '.join(f'{u}–{v}' for u, v in sorted(tuple(sorted(e)) for e in adj)) or 'no edges'}",
             f"v-structures: {', '.join(f'{u} → {c} ← {v}' for u, c, v in sorted(vs)) or 'none'}",
             f"equivalence class: {len(members)} DAG{'s' * (len(members) > 1)}",
             f"hypothesis {h}({a}, {b}) holds in {k} of {len(members)}",
             f"implied: {'yes' if k == len(members) else 'no'}"]
    return k == len(members), "\n".join(lines)


def _set(vs) -> str:
    return "{" + ", ".join(sorted(vs)) + "}"


def _paths(g: nx.DiGraph, a, b, z: set, limit: int = 6) -> tuple[list[str], bool]:
    """Open/blocked status per path (open paths first). (list of sentences, whether an open path exists)."""
    arrow = lambda p: "".join(f"{u} {'→' if g.has_edge(u, v) else '←'} " for u, v in zip(p, p[1:])) + p[-1]
    status = sorted(((p, blocked_by(g, p, z)) for p in paths_between(g, a, b)), key=lambda s: s[1] is not None)
    lines = [f"- {arrow(p)}: " + ("open" if r is None else f"blocked at {r[0]} ({'a collider not conditioned on, nor its descendants' if r[1] == 'collider' else 'a conditioned non-collider'})")
             for p, r in status[:limit]]
    lines += [f"- the remaining {len(status) - limit} paths are blocked"] if len(status) > limit and status[limit][1] is not None else []
    return lines or ["- no path connects them"], any(r is None for _, r in status)


def run_graph(args: dict) -> tuple[str, str]:
    """Execute one graph-structure query tool call. Returns (verdict, result sentence to give back to the model)."""
    names = [str(v) for v in args["variables"]]
    edges = []
    for e in args["graph"]:
        u, arrow, v = (x.strip() for x in str(e).partition("->"))
        if not arrow:
            raise ToolError(f"cannot read edge '{e}': use 'A -> B' with declared variables")
        if {u, v} - set(names):
            raise ToolError(f"cannot read edge '{e}': " + _undeclared([x for x in (u, v) if x not in names], names))
        edges.append((u, v))
    g = nx.DiGraph(edges)
    g.add_nodes_from(names)
    if not nx.is_directed_acyclic_graph(g):
        raise ToolError("the graph has a cycle")
    query, a, b = args["query"], str(args["a"]), str(args["b"])
    latent, given = [str(v) for v in args.get("unobserved", [])], [str(v) for v in args.get("given", [])]
    sets = [[str(v) for v in z] for z in args.get("sets", [])]
    bi = [tuple(x.strip() for x in re.split(r"<->|↔|\band\b", str(p))) for p in args.get("confounded", [])]  # CaLM CEI
    if any(len(p) != 2 or not set(p) <= set(names) or p[0] == p[1] for p in bi):
        raise ToolError("cannot read `confounded`: write each pair of declared variables with a shared unobserved confounder as 'A <-> B'")
    if bi and latent:
        raise ToolError("give unobserved confounders either as `unobserved` variables with their edges or as `confounded` pairs, not both")
    if query == "identify" and a == b and a in names:  # CaLM CEI contains items where treatment and outcome are the same
        return "identified", f"the treatment and the outcome are the same variable {a}: do({a}) sets it directly, so the effect is identified\nidentified: yes"
    if query not in GRAPH_QUERIES:  # mistake of calling det-cf through causal_graph (CLadder tool)
        raise ToolError(f"query must be one of {', '.join(GRAPH_QUERIES)}; a deterministic counterfactual with stated rules needs no tool: apply the rules by hand")
    if a == b or not {a, b, *latent, *given, *sum(sets, [])} <= set(names):
        raise ToolError(f"query must be one of {', '.join(GRAPH_QUERIES)}, with two different declared variables a and b and only declared variables elsewhere")
    for u, v in bi:  # for path checking, add an unobserved common-cause node for each confounded pair
        g.add_edges_from([(f"U[{u},{v}]", u), (f"U[{u},{v}]", v)])
        latent.append(f"U[{u},{v}]")
    if query in ("frontdoor", "instrument") and not sets:
        raise ToolError(f"{query} needs `sets`, e.g. [[\"M\"], [\"Z\"]]")
    if query == "frontdoor":  # CaLM FAS
        lines, verdicts = [], []
        for m in sets:
            cut = g.copy()
            cut.remove_nodes_from(m)
            ok = is_frontdoor(g, a, b, set(m))
            why = (f"contains {a} or {b}" if {a, b} & set(m) else
                   f"does not intercept the directed path {' → '.join(nx.shortest_path(cut, a, b))}" if nx.has_path(cut, a, b) else
                   "intercepts every directed path, and no back-door path opens from " + a + " to the set or from the set to " + b if ok else
                   f"intercepts every directed path, but a back-door path from {a} to the set or from the set to {b} (not blocked by {a}) stays open")
            verdicts.append("valid" if ok else "not valid")
            lines.append(f"set {_set(m)}: {verdicts[-1]} ({why})")
        return ", ".join(verdicts), "\n".join(lines)
    if query == "instrument":  # CaLM IV (a parent of the treatment that reaches the outcome only through the treatment; DoWhy definition)
        lines, verdicts = [f"parents (direct causes) of {a}: {_set(set(g.predecessors(a)) - set(latent))}"], []
        for z in sets:
            why = []
            for v in z:
                why += [f"{v} is the {'treatment' if v == a else 'outcome'} itself"] if v in (a, b) else [f"{v} is not a direct cause of {a}"] if not g.has_edge(v, a) else \
                    [f"{v} is connected to {b} other than through {a}"] if not is_instrument(g, a, b, v) else []
            verdicts.append("not valid" if why else "valid")
            lines.append(f"set {_set(z)}: {verdicts[-1]} ({'; '.join(why) if why else 'every variable is a direct cause of ' + a + ' with no path to ' + b + ' except through ' + a})")
        return ", ".join(verdicts), "\n".join(lines)
    if query == "path":  # CaLM AR/CA (parent/ancestor) used to be answered with dsep, reading 'not independent' as 'cause'
        parent, anc = g.has_edge(a, b), nx.has_path(g, a, b)
        lines = [f"parents of {b}: {_set(g.predecessors(b))}",
                 f"directed path from {a} to {b}: {' → '.join(nx.shortest_path(g, a, b)) if anc else 'none'}"]
        lines += [f"directed path from {b} to {a}: {' → '.join(nx.shortest_path(g, b, a))} ({b} is an ancestor of {a}, not the other way round)"] if nx.has_path(g, b, a) else []
        lines += [f"{a} is a parent (direct cause) of {b}: {'yes' if parent else 'no'}", f"{a} is an ancestor (direct or indirect cause) of {b}: {'yes' if anc else 'no'}"]
        return "parent" if parent else "ancestor" if anc else "none", "\n".join(lines)
    if query == "dsep":
        lines, open_ = _paths(g, a, b, set(given))
        verdict = "no" if open_ else "yes"
        return verdict, "\n".join([f"paths between {a} and {b} given {_set(given)}:", *lines, f"d-separated: {verdict}"])
    if query == "adjustment":
        if not sets:
            raise ToolError("adjustment needs `sets`, e.g. [[], [\"Z\"]]")
        fb, pb = forbidden(g, a, b), proper_backdoor_graph(g, a, b)
        lines, verdicts = [f"forbidden nodes ({a}, the nodes on causal paths from {a} to {b} and their descendants): {_set(fb)}"], []
        for z in sets:
            bad = sorted(set(z) & fb)
            body, open_ = _paths(pb, a, b, set(z), limit=4)
            ok = not bad and not open_
            verdicts.append("valid" if ok else "not valid")
            why = f"contains the forbidden {', '.join(bad)}" if bad else "a non-causal path stays open" if open_ else "no forbidden node, and every non-causal path is blocked"
            lines += [f"set {_set(z)}: {verdicts[-1]} ({why})"] + ([f"  {l}" for l in body if l.endswith("open")] if open_ and not bad else [])
        return ", ".join(verdicts), "\n".join(lines)
    e = identify(g, a, b, latent)
    if e.strategy == "no_effect":
        lines = [f"causal path from {a} to {b}: none, so the causal effect is zero"]
    else:
        lines = [f"causal path from {a} to {b}: {' → '.join(nx.shortest_path(g, a, b))}",
                 f"back-door: {'adjusting for the observed set ' + _set(e.variables) + ' is valid' if e.strategy == 'backdoor' else 'no set of observed variables is valid'}"]
        if e.strategy != "backdoor":
            fd = frontdoor_sets(g, a, b, latent)
            lines.append(f"front-door: {'the observed set ' + _set(fd[0]) + ' satisfies the front-door criterion' if fd else 'no observed mediator set satisfies the criterion'}")
            if not fd:
                iv = instruments(g, a, b, latent)
                lines.append(f"instrument: {iv[0] + ' is a valid instrument' if iv else 'no valid instrument'}")
    name = {"no_effect": "no causal effect", "backdoor": "back-door adjustment", "frontdoor": "front-door adjustment", "iv": "instrumental variable",
            "not_identified": "not identifiable from the observed variables"}[e.strategy]
    if bi:  # if confounded pairs exist, the ID algorithm (nonparametric, complete) decides. Instrumental variables assume linearity, so they do not count as identification
        obs = g.subgraph(names)
        ok = id_identifiable(obs, [frozenset(p) for p in bi], a, b)
        lines = [f"unobserved confounders: {', '.join(f'{u} <-> {v}' for u, v in bi)}"] + [l for l in lines if not l.startswith("instrument")]
        how = name if e.strategy in ("no_effect", "backdoor", "frontdoor") else "the ID algorithm (do-calculus)" if ok else "not identifiable (a hedge blocks every do-calculus derivation)"
        return "identified" if ok else "not_identified", "\n".join(lines + [f"general identification (ID algorithm): {how}", f"identified: {'yes' if ok else 'no'}"])
    return e.strategy, "\n".join(lines + [f"identified by: {name}"])


RUNNERS = {"causal_engine": run, "causal_discovery": run_discovery, "causal_graph": run_graph}


# ---------- Qwen3.5 tool-call format ----------
CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def format_call(args: dict, name: str = "causal_engine") -> str:
    """Same as how the chat template renders tool_calls (strings as is, everything else as JSON)."""
    body = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def parse_call(text: str) -> dict | None:
    """Arguments of the last tool call, or None if there is no call."""
    calls = CALL.findall(text)
    if not calls:
        return None
    out = {}
    for k, v in PARAM.findall(calls[-1][1]):
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v.strip()
    return out


AUTOFIX = True  # if the same name error occurs twice, read it as the closest declared name
REPEAT = "error: this call is identical to the previous call"
UNDECLARED = re.compile(r"(?:^|: )([^:()']+?) (?:is not a declared variable|are not declared variables) \(declared: ([^)]*)\)")


def _repeat(args: dict, error: str) -> str:
    """Error for re-sending the same arguments as a call that got an error. The same arguments give the same error, so the engine is not run again; the message points out which argument holds the wrong name."""
    msg = f"{REPEAT[7:]}, which failed ({error.removeprefix('error: ').split('; ')[0]}); the same arguments give the same error"
    fixes = []
    for names, declared in UNDECLARED.findall(error):
        for n in names.split(", "):
            where = [k for k, v in args.items() if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", json.dumps(v, ensure_ascii=False))]
            close = get_close_matches(n, declared.split(", "), n=1, cutoff=0)
            fixes.append(f"replace {n} with {close[0]} in {', '.join(where) or 'the call'}")
    return msg + ("; " + "; ".join(fixes) if fixes else "; change the argument the error points to")


def _distance(a: str, b: str) -> int:
    """Edit distance (one missing, extra or substituted character counts as 1)."""
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def _autofix(args: dict, error: str) -> tuple[dict, dict] | None:
    """If each name in a name error has exactly one closest declared name within edit distance 2, return (arguments with the names replaced, {wrong name: declared name})."""
    subs = {}
    for names, declared in UNDECLARED.findall(error):
        for n in names.split(", "):
            d = sorted((_distance(n, m), m) for m in declared.split(", "))
            if d[0][0] > 2 or len(d) > 1 and d[1][0] == d[0][0]:
                return None
            subs[n] = d[0][1]
    if not subs:
        return None
    pat = re.compile("|".join(rf"(?<!\w){re.escape(n)}(?!\w)" for n in sorted(subs, key=len, reverse=True)))
    return {k: json.loads(pat.sub(lambda m: subs[m.group()], json.dumps(v, ensure_ascii=False))) for k, v in args.items()}, subs


def _execute(name: str, args: dict, repair: bool = True) -> str:
    f = RUNNERS.get(name, run)
    try:
        return (f(args, repair=repair) if f is run else f(args))[1]
    except (ToolError, KeyError, TypeError, AttributeError, ValueError) as e:  # ValueError: numbers like '0.28.' in sampled output
        return f"error: {e}"


def respond(text: str, prev: str = "", autofix: bool = AUTOFIX, repair: bool = True) -> str:
    """Execute the tool call in the model output and return the result sentence. Errors are also returned as sentences so the model can fix them.

    prev: output before this call (calls and <tool_response> blocks). If the immediately preceding call was an error and this call is identical, the engine is not run again.
    For a name error with autofix, it reads the closest declared name, runs, and appends a note that it did so (_autofix). Otherwise it returns the _repeat error.
    """
    name, args = CALL.findall(text)[-1][0], parse_call(text)
    last = CALL.findall(prev)[-1:]
    error = prev.rsplit("<tool_response>\n", 1)[-1].split("\n</tool_response>")[0]
    if last and error.startswith("error") and (last[0][0], parse_call(prev)) == (name, args):
        fix = autofix and _autofix(args, error)
        if fix:
            return _execute(name, fix[0], repair) + f"\n(note: the same call failed twice, so the engine read {', '.join(f'{a} as {b}' for a, b in fix[1].items())})"
        return error if error.startswith(REPEAT) else f"error: {_repeat(args, error)}"
    return _execute(name, args, repair)
