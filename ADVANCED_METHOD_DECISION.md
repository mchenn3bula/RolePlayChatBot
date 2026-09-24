# Advanced-method decision: defer training

September 24, 2026. User request: try one advanced method only if justified.
**The present evidence does not justify launching an advanced-method run.**
This is an evidence assessment, not a failed experiment or a permission hold.
No new adapter, GPU training job, paid teacher call or model-default change was made.

## Evidence used

The completed, frozen [DPO comparison](DPO_COMPARISON.md) is the only local
SFT-versus-DPO quality evidence. Existing annotations were not changed or relabeled.
No source dialogue was read and no held-out test examples were accessed.

| Decision factor | Observation | Implication |
|---|---|---|
| Factual consistency | SFT and DPO both contradict 27/174 required facts | No measured reduction in the central failure |
| French consistency | Contradictions rise from 18 to 21 out of 87 | Bilingual improvement is not established |
| Completeness | Omissions fall from 61 to 49 | Useful pilot signal, but incomplete factual repair |
| Overall preference | DPO 27 wins, 67 ties, 14 losses | Modest assistant-reviewed improvement, not an independent human result |
| Training diversity | 160 rows from only 16 authored templates | More aliases/epochs would not add independent situations |
| Preference validation | 40 rows from four templates; raw chosen-over-rejected likelihood accuracy stays 87.5% | Reduced DPO loss alone does not establish improved generation |
| Attribution | No matched continued-SFT arm on the chosen responses | Extra examples/compute and objective effects remain confounded |
| Evaluation independence | Ten familiar scenario families; one training seed; reviewer authored curriculum | Weak basis for selecting a more complex optimizer |

The existing plan's provisional 20% relative contradiction-reduction target was
not met: on this fixed panel that would require at most 21 contradictions, versus
the observed 27. This is an engineering target, not a statistical significance
test. Passing it later would still require per-language and quality checks.

## One candidate retained: Se-DPO

[Se-DPO](https://arxiv.org/html/2608.09568v1) redistributes token credit using
policy/reference log-probability differences and reference entropy through a
learned calibration network. The paper explicitly treats its weighted objective
as an approximation; its assistant-benchmark results do not establish French
roleplay fidelity. Reviewed the method, objective and stated scope again for this
decision. The older [literature notes](LITERATURE_REVIEW.md) remain background.

Our hypothesis would be that localized wrong entities, objects and deadlines
benefit from token-specific credit. **We have not established that uniform token
weighting causes these errors, or that the learned credits identify factual
importance.** Current failures also include wrong speaker roles and stale scene
state. Narrow synthetic preferences and missing controls are plausible alternative
explanations, not proven diagnoses. A more elaborate loss cannot be assumed to
resolve them. Se-DPO therefore remains the sole candidate, not an authorized
automatic follow-on job. Distillation and RL are not being added to this experiment.

## Evidence that would justify reconsideration

First establish a stronger ordinary control: varied, user-reviewed EN/FR
preferences on natural model errors, with whole scenario families separated and
the evaluation set frozen before training. Retain correctness as distinct from
style; include fluent but factually wrong negatives. The current development
panel must remain identified as familiar if reused; using its examples for future
training would invalidate it as held-out evidence.

Compare the same S1 starting adapter, continued SFT on the chosen replies and
ordinary DPO. Keep the training budget explicitly matched and report both compute
and target-token exposure; keep evaluation messages, state and decoding identical.
Obtain independent preference review, especially for French. If residual errors
are reproducibly localized despite good preference data and a useful ordinary-DPO
control, freeze a Se-DPO-only protocol before examining its outputs.

That future pilot should initialize from S1 rather than stack onto D1, use the
exact S1 reference, and preserve the same data and trainable modules as its DPO
control. Resolve gradient flow from the paper/author implementation, test that
uniform credit recovers DPO, and measure memory including reference token entropy.
Retain the six-hour cap, visible technical logs, 18 GiB reservation ceiling and
4 GiB free-memory margin. Compare required-fact contradictions and omissions by
language; reject an apparent gain that merely becomes evasive or regresses French.
These are preparation criteria for a possible future experiment, not scheduled work.

## Scope of this update

Updated the research plan and handoff with this decision. Training code,
checkpoints, original comparison protocol and frozen annotations remain unchanged.
Only documentation and transfer-bundle membership changed; no model tests or GPU
preflight needed to be rerun. Aggregate counts and referenced local files were
checked against their saved artifacts.
