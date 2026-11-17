from collections import Counter

import numpy as np
import pytest

from hvme.gym.check import audit
from hvme.gym.dataset import _version, generate
from hvme.gym.formats import reward
from hvme.lm.tools import ToolError, format_call, parse_call, respond, run, run_graph
from hvme.lm.trajectories import tool_examples, transcript

CL = {"variables": {"u": ["1", "0"], "x": ["1", "0"], "m": ["1", "0"], "y": ["1", "0"]},
      "graph": ["u -> x", "u -> y", "x -> m", "m -> y"], "unobserved": ["u"],
      "probabilities": ["P(x=1) = 0.54", "P(x=0, y=1) = 0.27", "P(x=1, y=1) = 0.31"],
      "query": "correlation", "treatment": {"x": "1"}, "baseline": {"x": "0"}, "outcome": {"y": "1"}}


@pytest.fixture(scope="module")
def items():
    return generate(650, "train", seed=5, **_version(3)) + generate(260, "test", seed=5, **_version(3))


def test_tool_reads_non_cpt_statements():
    # P(y|x=1) − P(y|x=0) = 0.31/0.54 − 0.27/0.46
    assert run(CL)[0] == pytest.approx(0.31 / 0.54 - 0.27 / 0.46, abs=1e-4)


def test_tool_flags_unidentified_and_bad_input():
    ate = CL | {"query": "ate"}  # unobserved confounding, so not identified from the joint probabilities alone
    with pytest.raises(ToolError, match="not identifiable"):
        run(ate)


def test_tool_returns_sign_when_only_sign_is_identified():
    # frontdoor graph with no P(jyka) (a CLadder item): not point-identified, but the ATE is negative under every fit
    fd = {"variables": {"q": ["yes", "no"], "x": ["yes", "no"], "m": ["yes", "no"], "y": ["yes", "no"]},
          "graph": ["q -> x", "q -> y", "x -> m", "m -> y"],
          "probabilities": ["P(m=yes | x=no) = 0.51", "P(m=yes | x=yes) = 0.92", "P(y=yes | x=no, m=no) = 0.89", "P(y=yes | x=no, m=yes) = 0.46",
                            "P(y=yes | x=yes, m=no) = 0.50", "P(y=yes | x=yes, m=yes) = 0.18"],
          "query": "ate", "treatment": {"x": "yes"}, "baseline": {"x": "no"}, "outcome": {"y": "yes"}}
    val, out = run(fd)
    assert val < 0 and "not point-identified, but the sign is negative in every fit" in out


def test_tool_call_roundtrip():
    assert parse_call("풀이...\n\n" + format_call(CL)) == CL


def test_v3_generators_and_labels(items):
    assert set(Counter(d["query_type"] for d in items)) == set(_version(3)["generators"])
    assert audit(items) == []
    assert 0.4 < np.mean([d["label"] for d in items if d["kind"] == "noul"]) < 0.6


def test_v3_phrasing_varies(items):
    num = [d for d in items if "form" in d["meta"]]
    assert all(len({d["meta"]["form"] for d in num if d["query_type"] == t}) >= 2 for t in ("ate", "ett", "nie", "correlation"))
    assert {d["meta"]["q"]["treatment"][d["meta"]["x"]] for d in num if d["query_type"] == "ate"} == {0, 1}
    assert any(d["meta"]["minimal"] for d in num)


def test_tool_trajectories_are_correct(items):
    ex = tool_examples(items)
    assert all(reward(d, transcript(e["messages"])) == 1 for d, e in zip(items, ex))
    assert any(len(e["messages"]) == 4 for e in ex) and any(len(e["messages"]) == 2 for e in ex)


def test_tool_reads_cladder_style_states():
    bare = CL | {"probabilities": ["P(x) = 0.54", "P(not x and y) = 0.27", "P(x and y) = 0.31"]}
    assert run(bare)[0] == pytest.approx(run(CL)[0], abs=1e-6)


@pytest.fixture(scope="module")
def items31():
    return generate(500, "train", seed=6, **_version(31))


def test_v31_bare_items_and_repairs(items31):
    bare = [d for d in items31 if d["meta"].get("style") == "bare"]
    assert audit(items31) == [] and {d["query_type"] for d in bare} >= {"ate", "ett", "correlation", "det_cf"}
    assert all(d["lang"] == "en" and "not " in d["context"] + d["question"] for d in bare if d["query_type"] != "det_cf")
    ex = tool_examples(items31, repair=0.5, seed=1)
    assert {e["source"] for e in ex} == {"template", "repair"}
    assert all(reward(d, transcript(e["messages"])) == 1 for d, e in zip(items31, ex))
    assert all(e["messages"][2]["content"].startswith("error:") for e in ex if e["source"] == "repair")


def test_v32_named_items_use_first_states():
    items = generate(1200, "train", seed=7, **_version(32))
    named = [d for d in items if d["meta"].get("form") == "named"]
    assert audit(items) == [] and {d["query_type"] for d in named} == {"ate", "nde", "nie"}
    assert all(d["meta"]["q"]["outcome"][d["meta"]["y"]] == 1 and d["meta"]["q"]["treatment"][d["meta"]["x"]] == 1 for d in named)
    ex = tool_examples(named)
    assert all(reward(d, transcript(e["messages"])) == 1 for d, e in zip(named, ex))


def test_process_score_rewards_reasoning_not_luck(items):
    from hvme.lm.data import STRUCT, process
    from hvme.lm.solutions import solve

    for d in [d for d in items if d["query_type"] in STRUCT][:40]:
        ref = solve(d)
        lines = ref.splitlines()
        assert process(d, ref) == 1.0
        assert process(d, "\n".join(reversed(lines))) == 1.0  # line order is ignored
        i = next(k for k, l in enumerate(lines) if l.startswith("3)"))
        assert process(d, "\n".join(lines[:i] + [lines[-1]])) == 0.0  # 0 if only the answer is written without the steps after step 3


def test_discovery_tool_reads_corr2cause_premises():
    from hvme.lm.tools import run_discovery
    vs = {"variables": ["A", "B", "C"], "relations": ["A correlates with C.", "B correlates with C.", "A is independent of B."]}
    assert run_discovery(vs | {"hypothesis": "collider", "a": "A", "b": "B"})[0]  # v-structure A → C ← B
    assert run_discovery(vs | {"hypothesis": "direct", "a": "A", "b": "C"})[0]
    full = {"variables": ["A", "B", "C"], "relations": ["A correlates with B.", "A correlates with C.", "B correlates with C."]}
    implied, text = run_discovery(full | {"hypothesis": "collider", "a": "A", "b": "B"})
    assert not implied and "equivalence class: 6 DAGs" in text  # complete graph: true in only some of the 6
    chain = {"variables": ["A", "B", "C"], "relations": ["A correlates with B.", "B correlates with C.", "A correlates with C.", "A and C are independent given B."]}
    assert not run_discovery(chain | {"hypothesis": "via", "a": "A", "b": "C"})[0]  # direction is not determined
    ko = {"variables": ["A", "B", "C"], "relations": ["A와 C는 상관관계가 있다.", "B와 C는 상관관계가 있다.", "A와 B는 서로 독립이다."]}
    assert run_discovery(ko | {"hypothesis": "collider", "a": "A", "b": "B"})[0]
    with pytest.raises(ToolError, match="missing A–B"):
        run_discovery(vs | {"relations": vs["relations"][:2], "hypothesis": "direct", "a": "A", "b": "C"})
    with pytest.raises(ToolError, match="no DAG"):  # marginally independent but conditioning leaves no non-independent pair: collider C is in the conditioning set
        run_discovery(vs | {"relations": vs["relations"] + ["A and B are independent given C."], "hypothesis": "direct", "a": "A", "b": "C"})
    assert respond(format_call(vs | {"hypothesis": "collider", "a": "A", "b": "B"}, "causal_discovery")).endswith("implied: yes")


def test_via_matches_corr2cause_definition():
    import networkx as nx
    from hvme.engine.discovery import holds, mec_of
    g = nx.DiGraph([("A", "B"), ("B", "C"), ("A", "C")])
    assert holds("via", g, "A", "C") and holds("direct", g, "A", "C")  # true if a path through an intermediate variable exists, even with a direct edge
    assert len(mec_of(g)) == 6 and len(mec_of(nx.DiGraph([("A", "C"), ("B", "C")]))) == 1


def test_gym_v33_corr2cause_uses_discovery_tool():
    from hvme.lm.data import process
    items = [d for d in generate(400, "train", seed=7, **_version(33)) if d["query_type"] == "corr2cause"]
    assert items and not audit(items)
    assert max(len(d["meta"]["call"]["variables"]) for d in items) >= 5
    ex = tool_examples(items, repair=0.5)
    assert any(e["source"] == "repair" for e in ex)
    for d, e in zip(items, ex):
        tr = transcript(e["messages"])
        assert reward(d, tr) == 1 and process(d, tr) == 1.0 and "causal_discovery" in tr


def test_tool_choice_repair_switches_to_engine():
    items = [d for d in generate(600, "train", seed=3, **_version(32)) if d["meta"].get("form") == "named"]
    ex = [e for e in tool_examples(items, repair=1.0, seed=1) if e["messages"][1].get("train") is False]
    assert ex
    for e in ex:
        d = next(d for d in items if d["id"] == e["id"])
        first, tr = e["messages"][1], transcript(e["messages"])
        assert first["tool_calls"][0]["function"]["name"] == "causal_discovery" and e["messages"][2]["content"].startswith("error:")
        assert e["messages"][3]["tool_calls"][0]["function"]["name"] == "causal_engine" and reward(d, tr) == 1


def test_corr2cause_v33_cot_without_tools():
    from hvme.lm.trajectories import cot_examples
    items = [d for d in generate(400, "train", seed=7, **_version(33)) if d["query_type"] == "corr2cause" and len(d["meta"]["call"]["variables"]) <= 4]
    ex = cot_examples(items)
    assert ex and all(not e["tools"] and reward(d, e["messages"][1]["content"]) == 1 for d, e in zip(items, ex))


def test_digit_states_repair_and_repeat():
    # fix the mistake of declaring yes/no in a state 1/0 item. repeat fixes only after a turn that re-calls with the same arguments (excluded from the loss)
    items = [d for d in generate(300, "train", seed=4, **_version(34)) if d["meta"].get("call") and d["meta"]["call"]["variables"] == {v: ["1", "0"] for v in d["meta"]["call"]["variables"]}]
    ex = tool_examples(items, repair=1.0, seed=2, repeat=1.0)
    assert {e["source"] for e in ex} >= {"repeat"} and any(e["messages"][1]["tool_calls"][0]["function"]["arguments"]["variables"] != d["meta"]["call"]["variables"] for d, e in zip(items, ex))
    for d, e in zip(items, ex):
        if e["source"] != "repeat":
            continue
        ms = e["messages"]
        assert ms[3]["train"] is False and ms[3]["tool_calls"] == ms[1]["tool_calls"] and ms[2]["content"] == ms[4]["content"]
        assert ms[5]["tool_calls"][0]["function"]["arguments"] == d["meta"]["call"] and reward(d, transcript(ms)) == 1


def test_tool_choice_repair_only_named():
    items = generate(400, "train", seed=3, **_version(32))
    ex = tool_examples(items, repair=1.0, seed=1)
    switch = lambda e: e["messages"][1].get("train") is False and e["messages"][1]["tool_calls"][0]["function"]["name"] == "causal_discovery"
    assert all(d["meta"].get("form") == "named" for d, e in zip(items, ex) if switch(e))  # det_cf wrong-tool choice (causal_graph) is handled separately


def test_causal_graph_tool_matches_labels():
    # structural queries (dsep, backadj, identify) are causal_graph call trajectories, and the tool's verdict matches the Gym answer
    from hvme.lm.tools import run_graph
    from hvme.lm.trajectories import GRAPH, graph_call
    items = [d for d in generate(900, "train", seed=6, **_version(31)) if d["query_type"] in GRAPH]
    ex = tool_examples(items, repair=0.3, seed=3)
    assert {e["source"] for e in ex} == {"template", "repair"} and len({d["query_type"] for d in items}) == 3
    for d, e in zip(items, ex):
        assert e["messages"][-2]["role"] == "tool" and reward(d, transcript(e["messages"])) == 1
    assert run_graph(graph_call(next(d for d in items if "methods" in d["meta"])))[0] in ("valid, not valid", "not valid, valid")
    with pytest.raises(ToolError):
        run_graph({"variables": ["A", "B"], "graph": ["A -> C"], "query": "dsep", "a": "A", "b": "B"})


def test_named_flip_keeps_order_cue():
    # a named treatment comes after the baseline in conditional-probability sentences. Half swap the state names; present/absent-type states are not swapped
    from hvme.gym import v3
    items = generate(200, "train", seed=35, mix={k: 1 for k in v3.NAMED_PHRASES}, generators=_version(35)["generators"])
    flipped = 0
    for d in items:
        call = d["meta"]["call"]
        (x, tv), = call["treatment"].items()
        conds = " ".join(p.split("|", 1)[1] for p in call["probabilities"] if "|" in p)
        assert d["meta"]["form"] == "named" and conds.find(f"{x}={call['baseline'][x]}") < conds.find(f"{x}={tv}")
        node = next(k for k, w in d["meta"]["vocab"].items() if w[0 if d["lang"] == "ko" else 3] == x)
        w = d["meta"]["vocab"][node]
        flipped += w[4] in ("low", "normal", "slow", "poor", "unskilled", "off-peak")
        assert w[4] not in ("no", "off", "not taken", "skipped")
    assert 0 < flipped < len(items)
    ex = tool_examples(items, repair=0.3, seed=1)
    assert all(reward(d, transcript(e["messages"])) == 1 for d, e in zip(items, ex))


def test_state_spelling_and_error_names_states():
    # CLadder wrong answer: declared 'non-smoker' but wrote 'nonsmoker' in the probability sentence. Spelling differences are read as the same state, and unknown states get the declared states in the error
    call = CL | {"variables": {**CL["variables"], "x": ["smoker", "non-smoker"]},
                 "probabilities": ["P(x=Smoker) = 0.54", "P(x=nonsmoker, y=1) = 0.27", "P(x=smoker, y=1) = 0.31"],
                 "treatment": {"x": "smoker"}, "baseline": {"x": "non smoker"}}
    assert run(call)[0] == pytest.approx(run(CL)[0])
    with pytest.raises(ToolError, match="x was declared with the states smoker / non-smoker"):
        run(call | {"baseline": {"x": "no"}})


def test_named_flip_skips_digit_states():
    # 1/0 states are not swapped (CLadder has no item whose treatment is '0')
    from hvme.gym import v3
    items = generate(600, "test", seed=35, mix={k: 1 for k in v3.NAMED_PHRASES}, generators=_version(35)["generators"])
    assert all(next(iter(d["meta"]["call"]["treatment"].values())) != "0" for d in items)


def test_undeclared_state_repair(items):
    # first call uses an undeclared yes/no treatment state → error naming the declared states → corrected call
    from hvme.lm.trajectories import _repairs, tool_messages
    picked = [d for d in items if "undeclared" in _repairs(d["meta"])][:40]
    assert picked
    for d in picked:
        msgs = tool_messages(d, "undeclared")
        err = next(m["content"] for m in msgs if m["role"] == "tool")
        assert "was declared with the states" in err and reward(d, transcript(msgs)) == 1


def test_v29_repairs(items):
    # yes/no on the declaration side, a different outcome name, or the same condition twice gets an error naming the cause and is then fixed. det_cf is worked by hand after a wrong-tool choice
    from hvme.lm.trajectories import _repairs, messages
    want = {"declared": "was declared with the states yes / no", "alias": "is in no edge of the graph",
            "duplicate": "under the same condition", "detcf": "needs no tool"}
    for kind, text in want.items():
        picked = [d for d in items if d["query_type"] == "det_cf"][:10] if kind == "detcf" else [d for d in items if kind in _repairs(d["meta"])][:20]
        assert picked, kind
        for d in picked:
            msgs = messages(d, kind)
            assert text in next(m["content"] for m in msgs if m["role"] == "tool") and reward(d, transcript(msgs)) == 1, kind
    assert messages(picked[0], "detcf")[1]["train"] is False


def test_engine_names_duplicate_and_lone_variable():
    args = {"variables": {"x": ["yes", "no"], "y": ["yes", "no"], "y2": ["yes", "no"]}, "graph": ["x -> y"], "query": "ate",
            "probabilities": ["P(x=yes) = 0.5", "P(y=yes | x=yes) = 0.3", "P(y=yes | x=no) = 0.6"],
            "treatment": {"x": "yes"}, "baseline": {"x": "no"}, "outcome": {"y": "yes"}}
    assert run(args)[0] == pytest.approx(-0.3)  # variables without edges alone are not an error
    args["probabilities"] += ["P(y2=yes | x=yes) = 0.2", "P(y2=yes | x=no) = 0.7"]
    with pytest.raises(ToolError, match="inconsistent with the graph .*: y2 is declared but is in no edge"):
        run(args, repair=False)
    assert "the engine added x -> y2" in run(args)[1]  # adding the conditioning variable as a parent makes it correct
    args["variables"].pop("y2")
    del args["probabilities"][3:]
    args["probabilities"][2] = "P(y=yes | x=yes) = 0.6"
    with pytest.raises(ToolError, match="under the same condition"):
        run(args)
    args["probabilities"][2] = "P(y=yes | x=no) = 0.6"
    args |= {"query": "nde", "mediators": ["y"]}
    with pytest.raises(ToolError, match="leave the treatment and the outcome out"):
        run(args)


def test_engine_names_misspelled_variable():
    args = {"variables": {"x": ["1", "0"], "y": ["1", "0"], "fjq": ["1", "0"]}, "graph": ["x -> y", "y -> fjfq"], "query": "ate",
            "probabilities": ["P(x=1) = 0.5", "P(y=1 | x=1) = 0.3", "P(y=1 | x=0) = 0.6", "P(fjq=1 | y=1) = 0.2", "P(fjq=1 | y=0) = 0.4"],
            "treatment": {"x": "1"}, "baseline": {"x": "0"}, "outcome": {"y": "1"}}
    with pytest.raises(ToolError, match=r"fjfq is not a declared variable \(declared: x, y, fjq\)"):
        run(args)
    with pytest.raises(ToolError, match=r"'yy=1': yy is not a declared variable"):
        run({**args, "graph": ["x -> y", "y -> fjq"], "outcome": {"yy": "1"}})
    assert run({**args, "graph": ["x -> y", "y -> fjq"]})[0] == pytest.approx(-0.3)


def test_typo_repair_names_the_misspelled_variable():
    # nonsense-variable-name typo (in the declaration, a probability sentence or an edge) → the engine points at the name → corrected to the proper name
    import re
    from hvme.lm.trajectories import NUMERIC, typo_examples
    items = [d for d in generate(600, "train", seed=7, **_version(31)) if d["query_type"] in NUMERIC and "call" in d["meta"]
             and all(re.fullmatch(r"[a-z]{4,5}", v) for v in d["meta"]["call"]["variables"])][:80]
    ex = typo_examples(items, repeat=0.5)
    assert len(ex) == len(items) > 20 and {e["source"] for e in ex} == {"repair", "repeat"}
    for d, e in zip(items, ex):
        ms = e["messages"]
        assert re.search(r"is not a declared variable \(declared: ", ms[2]["content"]) and ms[1]["tool_calls"][0]["function"]["arguments"] != d["meta"]["call"]
        assert ms[-3]["tool_calls"][0]["function"]["arguments"] == d["meta"]["call"] and reward(d, transcript(ms)) == 1


def test_error_prefix_ends_after_engine_error_and_counts_repeats():
    # GRPO round 3: rollout prefix ending in a first call with an injected mistake plus the engine error (with repeat, also the re-call with the same arguments and the second error). Re-calls inside the prefix are not counted
    import re
    from hvme.lm.data import error_repeats
    from hvme.lm.trajectories import NUMERIC, error_prefix, tool_messages
    d = next(d for d in generate(300, "train", seed=7, **_version(31)) if d["query_type"] in NUMERIC and "call" in d["meta"]
             and all(re.fullmatch(r"[a-z]{4,5}", v) for v in d["meta"]["call"]["variables"]))
    one, two = error_prefix(d, "typo"), error_prefix(d, "typo", repeat=True)
    assert one.count("<tool_call>") == 1 and two.count("<tool_call>") == 2 and two.startswith(one)
    assert one.endswith("</tool_response>\n") and "is not a declared variable" in one
    assert error_repeats(two) == 1 and error_repeats(two, start=2) == 0
    fixed = transcript(tool_messages(d, "typo", repeat=True))
    again = two + two[two.rindex("<tool_call>"):two.rindex("</tool_call>") + 12] + "\n<tool_response>\nerror: x\n</tool_response>\n"
    assert error_repeats(fixed, start=2) == 0 and error_repeats(again, start=2) == 1


def test_repeat_after_error_points_to_the_misspelled_argument():
    # re-sending the same arguments after an error does not re-run the engine; it names the argument with the wrong name and the closest declared name
    args = {"variables": {"x": ["1", "0"], "y": ["1", "0"], "fenx": ["1", "0"]}, "graph": ["x -> y", "y -> fenx"], "query": "ate",
            "probabilities": ["P(x=1) = 0.5", "P(y=1 | x=1) = 0.3", "P(y=1 | x=0) = 0.6", "P(fenix=1 | y=1) = 0.2", "P(fenix=1 | y=0) = 0.4"],
            "treatment": {"x": "1"}, "baseline": {"x": "0"}, "outcome": {"y": "1"}}
    call = format_call(args)
    first = respond(call)
    assert first.startswith("error: cannot read 'fenix=1'")
    prev = call + f"\n<tool_response>\n{first}\n</tool_response>\n"
    second = respond(call, prev, autofix=False)
    assert second.startswith("error: this call is identical to the previous call") and second.endswith("replace fenix with fenx in probabilities")
    third = respond(call, prev + call + f"\n<tool_response>\n{second}\n</tool_response>\n", autofix=False)
    assert third == second  # the third gives the same error (not stacked)
    fixed = format_call(args | {"probabilities": [p.replace("fenix", "fenx") for p in args["probabilities"]]})
    assert not respond(fixed, prev).startswith("error")
    assert respond(call, call + "\n<tool_response>\nATE = 0.1\n</tool_response>\n") == first

    # auto-correction: the second identical call is read with the closest declared name, computed, and the reading is reported
    assert respond(call, prev) == respond(fixed) + "\n(note: the same call failed twice, so the engine read fenix as fenx)"
    # with two close names (fenx, fenik), no correction is made
    amb = args | {"variables": args["variables"] | {"fenik": ["1", "0"]}, "graph": args["graph"] + ["y -> fenik"]}
    amb_call = format_call(amb)
    assert respond(amb_call, amb_call + f"\n<tool_response>\n{respond(amb_call)}\n</tool_response>\n").startswith("error: this call is identical")


def test_exp_away_items_and_given_condition():
    # comparison restricted to C=c in X → C ← Y. The engine uses correlation + given, and the trajectory puts the common effect in given
    from hvme.gym import v3
    from hvme.lm.trajectories import tool_examples
    items = generate(60, "train", seed=36, mix={"exp_away": 1}, generators=v3.GENERATORS_36)
    assert len(items) == 60 and audit(items) == [] and {d["meta"]["style"] for d in items} == {"bare", "cl", "gym"}
    ex = tool_examples(items, repair=0.3, seed=36)
    assert all(reward(d, transcript(e["messages"])) == 1 for d, e in zip(items, ex))
    call = items[0]["meta"]["call"]
    assert call["query"] == "correlation" and len(call["given"]) == 1
    no_given = {k: v for k, v in call.items() if k != "given"}
    assert abs(run(call)[0] - items[0]["value"]) < 0.01 and abs(run(no_given)[0]) < 1e-3  # without filtering on the common effect, the two causes are independent
    with pytest.raises(ToolError, match="`given` is only for correlation"):
        run(call | {"query": "ate"})


def test_engine_v212_pn_ps_ett_and_mediators():
    # CauGym item answers (PN upper bound 0.4735, PS lower bound 0.6386, ETT 0.0971) and NDE/NIE that are 0 because there is no path (no probabilities, mediator set automatic)
    from hvme.lm.tools import run
    V = lambda *n: {k: ["high", "low"] for k in n}
    pn = {"variables": V("e", "i", "j", "h"), "graph": ["e -> i", "e -> j", "i -> j", "j -> h"], "query": "pn", "bound": "upper",
          "probabilities": ["P(j=low | i=low, e=low) = 0.4412", "P(e=low) = 0.6272", "P(j=low | i=low, e=high) = 0.3416", "P(e=high) = 0.3728",
                            "P(i=low, j=low) = 0.1636", "P(i=high, j=high) = 0.5079"],
          "treatment": {"i": "high"}, "baseline": {"i": "low"}, "outcome": {"j": "high"}}
    assert abs(run(pn)[0] - 0.4735) < 5e-4
    ps = {"variables": V("h", "z", "a", "s"), "graph": ["h -> z", "z -> a", "z -> s", "a -> s"], "query": "ps", "bound": "lower",
          "probabilities": ["P(s=low | z=high) = 0.1475", "P(s=low) = 0.3184", "P(z=low, s=low) = 0.2676"],
          "treatment": {"z": "high"}, "baseline": {"z": "low"}, "outcome": {"s": "high"}}
    assert abs(run(ps)[0] - 0.6386) < 5e-4
    ett = {"variables": V("u", "n", "f", "z", "o"), "graph": ["u -> n", "u -> o", "n -> f", "n -> z", "f -> z", "f -> o", "z -> o"], "query": "ett",
           "probabilities": ["P(o=low | n=high) = 0.5597", "P(o=low | u=low, n=low) = 0.5136", "P(u=low | n=high) = 0.7773",
                             "P(o=low | u=high, n=low) = 0.2848", "P(u=high | n=high) = 0.2227"],
           "treatment": {"n": "low"}, "baseline": {"n": "high"}, "outcome": {"o": "low"}}
    val, text = run(ett)
    assert abs(val + 0.0971) < 5e-4 and "= +0.0971 (the effect of treatment on the treated" in text
    g = {"variables": V("t", "e", "j", "r", "w"), "graph": ["t -> e", "t -> w", "e -> j", "e -> r", "j -> r", "j -> w", "r -> w"], "probabilities": []}
    val, text = run(g | {"query": "nde", "treatment": {"e": "low"}, "baseline": {"e": "high"}, "outcome": {"w": "low"}})
    assert val == 0 and "path from the treatment to the outcome): j, r" in text and "graph alone" in text
    val, text = run(g | {"query": "nie", "treatment": {"t": "high"}, "baseline": {"t": "low"}, "outcome": {"e": "low"}})
    assert val == 0 and "no mediators" in text


def test_engine_not_state():
    # CLadder: 'gender=not male' is read as whichever of the two states is not male
    from hvme.lm.tools import _assign
    s = {"gender": ["male", "female"], "y": ["yes", "no"]}
    assert _assign("gender=not male, y=yes", s) == _assign("gender=female, y=yes", s) == {"gender": 0, "y": 1}
    with pytest.raises(ToolError):
        _assign("gender=not tall", s)


def test_graph_path_query():
    """path judges parents and ancestors along arrow direction and flags reversed paths."""
    c = {"variables": list("ABCDEF"), "graph": ["A -> E", "A -> D", "A -> F", "B -> D", "B -> F", "C -> E", "D -> E"], "query": "path"}
    assert run_graph(c | {"a": "A", "b": "E"})[0] == "parent"
    assert run_graph(c | {"a": "B", "b": "E"})[0] == "ancestor"
    v, text = run_graph(c | {"a": "E", "b": "A"})
    assert v == "none" and "A is an ancestor of E" in text
    assert run_graph(c | {"a": "E", "b": "F"})[0] == "none"


def test_ancestry_items():
    from hvme.gym import ancestry
    from hvme.gym.check import audit
    from hvme.gym.dataset import generate
    from hvme.gym.formats import reward
    from hvme.lm.trajectories import tool_examples, transcript
    items = generate(60, "train", seed=38, mix={f"ancestry_{k}": 1 for k in ancestry.ASKS}, generators=ancestry.GENERATORS_38, langs=("en",))
    assert len(items) == 60 and not audit(items)
    assert all(reward(d, transcript(e["messages"])) == 1 for d, e in zip(items, tool_examples(items)))


def test_graph_frontdoor_instrument_confounded():
    """frontdoor and instrument are judged per option set, and identify uses the ID algorithm when confounded is present."""
    c = {"variables": list("ZXMY"), "graph": ["Z -> X", "X -> M", "M -> Y"], "a": "X", "b": "Y", "sets": [["M"], ["Z"], ["Y"]]}
    assert run_graph(c | {"query": "frontdoor"})[0] == "valid, not valid, not valid"
    assert run_graph(c | {"query": "instrument"})[0] == "not valid, valid, not valid"
    assert run_graph(c | {"query": "frontdoor", "confounded": ["X <-> Y"]})[0].startswith("valid")
    ident = lambda conf: run_graph({k: v for k, v in c.items() if k != "sets"} | {"query": "identify", "confounded": conf})[0]
    assert ident(["X <-> Y"]) == "identified"  # frontdoor
    assert ident(["Z <-> X"]) == "identified"
    assert ident(["X <-> M"]) == "not_identified"  # a bow: confounding overlapping the X → M edge
    assert ident(["X and Y", "M <-> Y"]) == "not_identified"  # X-Y and M-Y confounding blocks the frontdoor
    bow = {"variables": ["X", "Y"], "graph": ["X -> Y"], "query": "identify", "a": "X", "b": "Y", "confounded": ["X <-> Y"]}
    assert run_graph(bow)[0] == "not_identified" and run_graph(bow | {"b": "X"})[0] == "identified"
    with pytest.raises(ToolError):
        run_graph(bow | {"confounded": ["X <-> Q"]})


def test_graphid_items():
    from hvme.gym import graphid
    from hvme.gym.check import audit
    from hvme.gym.dataset import generate
    from hvme.gym.formats import reward
    from hvme.lm.trajectories import tool_examples, transcript
    items = generate(60, "train", seed=39, mix={f"graphid_{k}": 1 for k in graphid.ASKS}, generators=graphid.GENERATORS_39, langs=("en",))
    assert len(items) == 60 and not audit(items)
    assert all(reward(d, transcript(e["messages"])) == 1 for d, e in zip(items, tool_examples(items)))


def test_engine_repair_states_and_implied_edges():
    """An invented declared state (high / normal) is read as the state used in the call (high / low), and a conditional probability that depends on a non-parent variable is computed by adding that edge."""
    args = {"variables": {"w": ["high", "normal"], "y": ["yes", "no"]}, "graph": ["w -> y"], "query": "ate",
            "probabilities": ["P(w=high) = 0.5", "P(y=yes | w=high) = 0.3", "P(y=yes | w=low) = 0.6"],
            "treatment": {"w": "low"}, "baseline": {"w": "high"}, "outcome": {"y": "yes"}}
    val, out = run(args)
    assert val == pytest.approx(0.3) and "read its states as high / low" in out
    with pytest.raises(ToolError, match="declared with the states high / normal"):
        run(args, repair=False)
    args["probabilities"].append("P(y=yes | w=normal) = 0.1")  # no correction if three states were used
    with pytest.raises(ToolError):
        run(args)
    med = {"variables": {"x": ["1", "0"], "m": ["1", "0"], "z": ["1", "0"], "y": ["1", "0"]}, "graph": ["z -> x", "x -> m", "m -> y", "z -> y"], "query": "ate",
           "probabilities": ["P(z=1) = 0.4", "P(x=1 | z=1) = 0.7", "P(x=1 | z=0) = 0.2", "P(m=1 | x=1, z=1) = 0.8", "P(m=1 | x=0, z=1) = 0.3",
                             "P(m=1 | x=1, z=0) = 0.6", "P(m=1 | x=0, z=0) = 0.1", "P(y=1 | m=1, z=1) = 0.5", "P(y=1 | m=0, z=1) = 0.4",
                             "P(y=1 | m=1, z=0) = 0.9", "P(y=1 | m=0, z=0) = 0.2"],
           "treatment": {"x": "1"}, "baseline": {"x": "0"}, "outcome": {"y": "1"}}
    with pytest.raises(ToolError, match="inconsistent"):
        run(med, repair=False)
    val, out = run(med)  # graph with z → m added: ATE = Σ_z P(z) Σ_m P(y|m,z)(P(m|x=1,z) − P(m|x=0,z))
    assert "the engine added z -> m" in out and val == pytest.approx(0.4 * 0.1 * 0.5 + 0.6 * 0.7 * 0.5)


def test_parse_relation_korean_variants():
    """Korean conditions are read however the model rewrites them ('를 주었을 때', 'given B일 때', condition after the pair)."""
    from hvme.lm.tools import parse_relation
    n = list("ABCDEF")
    for s, pair, given in [("B 를 주었을 때 A 와 D 는 독립적이다.", "AD", "B"), ("given B일 때 A와 C는 독립적이다.", "AC", "B"),
                           ("given B, A와 C는 독립적이다.", "AC", "B"), ("A 와 B 를 주었을 때 C 와 E 는 독립적이다.", "CE", "AB"),
                           ("A와 C는 B, D를 주었을 때 독립적이다.", "AC", "BD"), ("C 가 주어졌을 때 A 와 B 는 독립이다.", "AB", "C")]:
        assert parse_relation(s, n) == (frozenset(pair), frozenset(given)), s
    assert parse_relation("A 는 B 와 상관관계가 있다.", n) == (frozenset("AB"), None)
