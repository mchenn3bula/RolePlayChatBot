# Response-rule refinement: mixed result, no promotion

The extra first-person/completeness rule improved coverage on fresh authored
scenes, but regressed on familiar scenes. It fails the frozen acceptance gate.
Keep the default unchanged. Natural-reference prompting remains an experimental
candidate; neither prompting intervention establishes reliable role tracking.
No training or checkpoint modification was performed.

## What was compared

`RESPONSE_REFINEMENT_PROTOCOL.md` was frozen before generation. The `natural`
arm uses the preceding natural-reference prompts. The `direct` arm inserts one
sentence asking the character to speak in first person, answer every part of the
question using current facts, and correct false premises. Facts, their order,
history and user messages are otherwise identical. This bundles three requests;
the experiment cannot identify their individual effects. The earlier literature
rationale remains in `NATURAL_REFERENCE_RESEARCH.md`.

Both arms use Ministral 3 3B at revision
`b6d637bef2393152b3da2b2fde72eecdee30557e`, the frozen
`checkpoints/ministral-v2-csft-v1/epoch-02` adapter, native tokenizer, FP16/SDPA,
2,048-token context, 192-token output limit, temperature 0.7, top-p 0.9, top-k 50,
repetition penalty 1 and seeds 42/43/44. Generation order alternates arms.

There are 48 familiar requests and 36 fresh requests: six new scene families,
each in English and French with three seeds. Both panels were frozen before any
new outputs. Fresh scenes cover a returned pencil, supervisor/guide authority,
an appointment change, uncertain seed counts, library-copy restrictions, and a
puppet demonstration. They are new instances of related skills, not an
independent benchmark or the reserved project test split. Their difficulty differs
from the familiar panel, so compare arms within each panel, not absolute rates
across panels. Private source conversations were not read.

## Paired results

Counts below are replies; errors can overlap. A semantic pass means no role,
contradiction, omission or agency error and the requested language is correct.
It does not require perfect style or independent human approval.

| Measure | Familiar natural | Familiar direct | Fresh natural | Fresh direct |
|---|---:|---:|---:|---:|
| Replies | 48 | 48 | 36 | 36 |
| Semantic passes | 41 | 39 | 10 | 14 |
| Omissions | 7 | 8 | 21 | 17 |
| Role-attribution errors | 0 | 1 | 5 | 5 |
| Contradictions | 0 | 1 | 6 | 5 |
| Third-person self-description | 6 | 2 | 13 | 12 |
| Semantic and voice passes | 35 | 37 | 7 | 10 |
| Agency errors | 0 | 0 | 0 | 0 |
| Correct language | 48 | 48 | 36 | 36 |

Familiar English semantic passes fell 21 to 20/24; French fell 20 to 19/24.
Fresh English improved 8 to 9/18; French improved 2 to 5/18, still a low rate.
Paired preferences: familiar direct 8 wins, natural 6, 34 ties; fresh direct 6,
natural 2, 28 ties. Preference and strict semantic pass measure different things.

Both fresh language gates pass, but both familiar language gates fail, as does
the requirement that familiar omissions strictly decrease. Improved voice or
pooled totals cannot rescue these failures. This gate is a conservative local
decision rule, not a significance test. Three seeds from one scene are correlated.

## What the replies reveal

- The new rule reduces familiar role recitation, but sometimes removes required
  information. One ceramics answer correctly says who fires the kiln yet drops
  the correction that Faye is a partner rather than a daughter.
- Telescope French seed 42 introduces a new attribution error: the reply assigns
  the eyepiece case to Paz, although the speaking character carries it.
- The fresh concert handoff fails all six requests under both arms. Five replies
  assign the returned pencil to the user; the sixth omits it. An explicit current
  fact and a generic instruction are insufficient to reliably override history.
- Library constraints improve from one to three semantic passes out of six.
  Seed-count and puppet families each improve from two to three. Appointment
  updates remain at two, and museum authority remains at three.
- Zero role errors on the preceding familiar natural-reference panel did not
  generalize: that same arm has five on these new scenes. Correct third-person
  role recitation is a separate voice problem, not automatically a contradiction.

## Review and verification

All 168 replies were saved and reviewed in 84 randomized A/B pairs. The 48
familiar natural outputs reproduced their archived output tokens exactly.
Seventy-two reply ratings reused frozen historical item/text anchors; 96 were
new ratings. Historical eight-field scores were preserved; a ninth field records
explicit third-person self-description. New judgments and preferences were
frozen before unblinding. Identical item/text replies require identical scores
and a tie. Review remains single-assistant and non-independent; the author also
wrote the fresh scenes, and familiar replies limit blinding.

All replies ended with EOS, with no empty outputs or length-cap hits. Mean
input tokens natural/direct: familiar 367.06/401.06, fresh 388.42/422.42. Mean
output tokens: familiar 22.44/21.29, fresh 27.31/28.44. Mean repeated four-gram
fraction: familiar 0.00932/0.00544, fresh 0/0.00231. These short-output repetition
metrics do not establish scene quality.

GPU preflight had finite logits and all model parameters frozen. Native serving
inputs were verified for all 168 replies. Peak reserved VRAM was 9.22 GiB;
11.90 GiB was free at the final measurement, not a continuous minimum. The suite
ran 81 tests: 80 passed and one native-only test skipped in the baseline
environment. Ruff and launcher syntax checks passed. No GPU job remains active.

Artifacts remain together in `reports/ministral-response-refinement-v1/`:
`comparison.html`, frozen requests/protocol/anchors, generations, manifests,
randomized review and key, `assistant_review.jsonl`, `review_frozen.json`, and
`summary.json`. The compact aggregate is
`results/ministral-response-refinement-v1.json`; technical logs are under
`reports/logs/ministral-response-refinement-v1-*.log`, and the test log is
`reports/response-refinement-tests.log`. Generated dialogue stays out of Git.

## Next refinement

Do not adopt the extra global rule as a blanket upgrade. First isolate the
stale-update failure with separately frozen, authored EN/FR handoff scenes:
compare the obsolete historical possession exchange present versus absent,
keeping authoritative current-state content and voice instructions identical.
This is a diagnostic ablation, not a proposal to discard dialogue in production.
Preserve the present results as diagnostics and reserve new scene families before
generation. If evidence then supports a training intervention,
consider a separately scoped bilingual supervision experiment for state updates
and complete answers, with disjoint training/validation/evaluation scenes.
Neither follow-up has started; advanced optimization remains deferred.
