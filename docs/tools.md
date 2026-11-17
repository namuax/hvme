# LM tools

The LM never computes a causal quantity itself. It reads the passage, writes the variables, graph and probabilities as tool arguments, and answers from the tool's result. Tool definitions are in `humm.lm.tools.TOOLS` (OpenAI function format) and the loop that runs them is `humm.lm.sample.generate_tools`.

| Tool | Use | Main arguments |
|---|---|---|
| `causal_engine` | numeric queries on a binary SCM | `variables` (name → states), `graph` (`"A -> B"`), `probabilities` (`"P(B=yes \| A=no) = 0.3"`), `query` (`marginal`, `correlation`, `ate`, `cde`, `ett`, `nde`, `nie`, `pn`, `ps`), `treatment`, `baseline`, `outcome`, `mediators`, `given`, `bound` |
| `causal_graph` | structural queries that need no probabilities | `graph`, `confounded` (`"A <-> B"`), `query` (`path`, `dsep`, `adjustment`, `frontdoor`, `instrument`, `identify`), `a`, `b`, `given` |
| `causal_discovery` | what a set of (in)dependence statements implies | `variables`, `relations` ("A is independent of B given C"), `hypothesis` (`direct`, `via`, `cause`, `collider`, `confounder`), `a`, `b` |

## A call and its result

```
<tool_call>
<function=causal_engine>
<parameter=variables>
{"smoking": ["smoker", "nonsmoker"], "tar": ["high", "low"], "cancer": ["yes", "no"]}
</parameter>
<parameter=graph>
["smoking -> tar", "tar -> cancer"]
</parameter>
<parameter=probabilities>
["P(smoking=smoker) = 0.4", "P(tar=high | smoking=smoker) = 0.9", "P(tar=high | smoking=nonsmoker) = 0.2",
 "P(cancer=yes | tar=high) = 0.3", "P(cancer=yes | tar=low) = 0.1"]
</parameter>
<parameter=query>
ate
</parameter>
<parameter=treatment>
{"smoking": "smoker"}
</parameter>
<parameter=baseline>
{"smoking": "nonsmoker"}
</parameter>
<parameter=outcome>
{"cancer": "yes"}
</parameter>
</function>
</tool_call>
<tool_response>
P(cancer=yes | do(smoking=smoker)) = 0.2800
P(cancer=yes | do(smoking=nonsmoker)) = 0.1400
value = P(cancer=yes | do(smoking=smoker)) − P(cancer=yes | do(smoking=nonsmoker)) = +0.1400
</tool_response>
```

## Error messages are part of the design

When a call cannot be evaluated, the engine answers with an error the model can act on, and the training data contains trajectories that recover from each kind:

- an undeclared variable or state (the message lists the declared names and states),
- a statement that does not name two declared variables,
- a probability table that does not determine the quantity (the message gives the identifiable range, or its sign when the range is narrow),
- a repeated identical call.

The harness also fixes a few mistakes without the model's help: a name that fails twice is read as the closest declared name (edit distance ≤ 2), a state written in a different spelling is matched case-, space- and hyphen-insensitively, and `not <state>` of a two-state variable is read as the other state.
