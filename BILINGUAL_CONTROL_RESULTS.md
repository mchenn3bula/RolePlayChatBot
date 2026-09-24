# Bilingual preferences and the continued-SFT control

September 24, 2026. **The continued-SFT control is the stronger candidate in this
pilot. Standard DPO did not add value over the same chosen-response supervision.**
Dataset curation, both training runs and the frozen three-arm evaluation are
complete. This remains small, non-independent assistant-reviewed evidence.
The protocol is
[`BILINGUAL_CONTROL_PROTOCOL.md`](BILINGUAL_CONTROL_PROTOCOL.md).

## Completed reply comparison

| Measurement | Unchanged S1 | Continued SFT | Standard DPO |
|---|---:|---:|---:|
| All required constraints satisfied / 48 | 25 | **29** | 24 |
| EN constraints satisfied / 24 | 15 | 15 | 14 |
| FR constraints satisfied / 24 | 10 | **14** | 10 |
| Replies with a contradiction / 48 | 15 | 15 | 20 |
| Replies with an omission / 48 | 11 | **5** | 7 |
| EN replies with a contradiction / 24 | 6 | 7 | 8 |
| FR replies with a contradiction / 24 | 9 | **8** | 12 |
| Coherence / 5 | 3.52 | **3.75** | 3.52 |
| Relevance / 5 | 3.88 | **4.08** | 3.94 |
| Grammar / 5 | 4.81 | 4.92 | **5.00** |
| User-agency violations / 48 | 2 | 2 | 2 |
| Requested language followed / 48 | 48 | 48 | 48 |

Direct preference between trained arms: **continued SFT wins 7, DPO wins 1,
40 ties**. English: SFT 2 / DPO 0 / ties 22. French: SFT 5 / DPO 1 / ties 18.
The reviewer ranked randomized A/B/C outputs before the model key was opened.
Annotations were frozen and hashed; the assistant also authored the curriculum.

These are response-level flags, not the old D1 report's required-fact slot counts;
the panels differ, so the numbers must not be compared as a longitudinal gain.
A reply may contain both an omission and a contradiction. All constraints passed
requires neither, no agency violation, and correct language. A short reply is
accepted when it conveys the required facts; optional elaboration is not mandatory.
The minimum semantic criteria were clarified after training but before any
evaluation output was read, in `reports/ministral-bilingual-control-v2/review_criteria.json`.
The original protocol and expected responses were not edited. Borderline decisions
are documented in each annotation.

With ties as half wins, DPO's share against continued SFT is 43.75%. The descriptive
10,000-resample bootstrap over eight scene families yields 41.67–46.88%. This does
not cover reviewer bias, alternative data curation, other training seeds or
generalization beyond these familiar skill patterns. It is not proof of universal
SFT superiority. All three arms produced identical text on 20 requests.

No arm produced empty or capped evaluation replies; all 48 replies per arm ended
on native EOS. Mean output tokens were 22.96 / 20.94 / 22.85 for S1/SFT/DPO. Mean
within-reply repeated word 4-gram fractions were 0.152% / 0% / 0.056%; short outputs
and these low repetition rates do not establish correctness. Generation took
53.9 / 50.1 / 53.4 seconds with 9.22 GiB peak reservation in each arm.

## Improvements, regressions and remaining errors

- `r2-telescope_cover-fr-s42`: continued SFT correctly leaves the cover with Paz
  and the eyepiece case with the speaker. DPO incorrectly places the cover in the
  user's hand.
- `r2-ceramics_partner-fr-s44`: continued SFT corrects daughter to business partner
  and identifies itself as kiln operator. DPO transfers kiln operation to the user.
- `r2-painting_collection-en-s44`: continued SFT keeps the paid painting at the
  workshop awaiting collection. DPO unnecessarily claims it cannot verify the
  location and must ask the user.
- DPO's one preference win, `r2-song_sharing-fr-s44`: it offers private work on the
  song; continued SFT wanders into suggesting listening to silence. Both preserve
  user choice, but DPO is more helpful here.
- Negative outcome, `r2-song_sharing-en-s43`: unchanged S1 suggests sharing with a
  friend. Both trained arms are worse: continued SFT gives a vague songwriting
  offer, while DPO restarts lyrics that already exist.
- Shared failure: all three often copy second-person scene wording into replies,
  transferring the character's role, knowledge or possessions to the user. The new
  scenes use P1's schema but include prose scene facts; this experiment does not
  isolate atomic entity-based state wording from prose wording.

Continue with the simpler SFT candidate for further evaluation, but do not call
it a reliable upgrade: overall contradictions are unchanged from S1, and English
contradictions rise by one. A controlled state-wording ablation using explicit
character/user entity names is now a more specific next hypothesis than changing
the optimizer. Independent human review, broader natural errors and new scene
families remain valuable. Advanced methods stay deferred and the chat default
has not changed.

## Stronger preference data

The revised dataset has **40 training scene families**, up from D1's 16. Repeated
character aliases were removed; there is exactly one English and one French
pair per scene. The new scenes vary actual facts and tasks, while retaining
similar skill patterns. Forty scene instances do not imply forty independent
reasoning mechanisms. There are 80 training pairs, 24 validation pairs from 12
other families, and 16 new evaluation prompts from eight further families.
Translations remain in the same split. The locked final test remains unused.

The curriculum targets current ownership, updated dates/locations, relationships,
unknown information, speaker roles and user agency. Some prompts include authored
older dialogue that current P1 state explicitly supersedes. The same history and
state are supplied to each evaluation arm.

S1 generated one candidate on each of the 64 new train/validation prompts. All
were inspected: 32 clear errors became rejected answers (24 training, eight
validation); 32 correct or ambiguous alternatives were not assigned negative
labels. Authored contrasts fill those pairs instead. The training natural errors
include 11 EN / 13 FR replies. Actual errors include role reversals, stale state,
unfounded claims, evasions and one capped internal-state JSON dump. Its capped
text is an explicitly undesirable negative, not a complete successful reply.

The chosen responses are authored corrections. Judgments are **assistant-curated,
not independently human-labeled**. Some choices involve completeness rather than
direct falsehood, and a human may disagree with these borderline preferences.
The saved reasons make them auditable. Review all candidates and the chosen
replacements in `reports/ministral-preferences-v2-curation/review.html`.

Per training epoch there are 1,686 chosen target tokens and 1,934 rejected target
tokens (including EOS), averaging 21.08 versus 24.18 per reply. Length and style
therefore remain potential preference cues. No claim of a length-controlled
dataset is made. Natural negatives are mixed with authored contrasts, not a
fully on-policy preference corpus. No private source dialogue was inspected.

## The matched simpler control

All three arms use the immutable S1 adapter as their common reference point:

| Arm | Starting checkpoint | Additional training |
|---|---|---|
| baseline | S1 epoch-02 | None |
| csft | S1 epoch-02 | Reply-only SFT on the 80 chosen responses |
| dpo | S1 epoch-02 | Standard sigmoid DPO on the corresponding 80 pairs |

The trained arms share rank-16 attention q/k/v/o LoRA, FP32 adapters, frozen FP16
backbone, zero dropout, seed 20260928, shuffled example order, accumulation 8,
peak LR 5e-6 with 10% warmup/cosine decay, and two epochs. Both use the final
epoch-02 checkpoint by prior agreement: **20 updates and 160 chosen presentations**.
Checkpoint selection did not use generated evaluation results.

This matches positive examples, exposure and updates, **not FLOPs or wall time**.
DPO processes rejected replies as well, with beta 0.1 and exact frozen S1
reference scores. Both runs calculate reference-based validation diagnostics;
the SFT optimization path receives only chosen-response labels. Per-example
weighting also differs conventionally: token-weighted SFT CE versus equal-pair
DPO. Results apply to this fixed shared-LR pilot, not separately optimized recipes.

The new evaluation has 48 requests: 16 EN/FR prompts times three sampling seeds.
All arms receive the same native input IDs, authored history/state, temperature
0.7, top-p 0.9, top-k 50, repetition penalty 1.0 and 192-token output cap. The
three-way gallery is `reports/ministral-bilingual-control-v2/comparison.html`.
The model mapping is concealed during assistant A/B/C review. The reviewer also
authored/curated the dataset, so this is not an independent or human study.

Machine-readable measurements are in `results/ministral-bilingual-control-v2.json`.
Complete annotations, criteria, hashes and all generated replies remain under
`reports/ministral-bilingual-control-v2/`.

## Verification and artifacts

| Training measurement | Continued SFT | Standard DPO |
|---|---:|---:|
| Optimizer updates | 20 | 20 |
| Chosen-response presentations | 160 | 160 |
| Chosen target tokens presented | 3,372 | 3,372 |
| Rejected target tokens in optimization | 0 | 3,868 |
| Full sequence tokens processed in optimization | 59,448 | 119,392 |
| Optimization time | 41.4 s | 81.9 s |
| Total run time, including loading/reference/validation/saves | 96.3 s | 133.3 s |
| Peak reserved VRAM | 9.17 GiB | 9.17 GiB |
| Overflow retries in real training | 0 | 0 |
| Final held-out chosen NLL, token-weighted | 1.5213 | 1.5548 |
| Final held-out DPO preference loss | 0.6414 | 0.6414 |

Both start at chosen NLL 1.6089 and preference loss 0.6931. Raw chosen-over-rejected
sequence-likelihood accuracy stays 62.5% in both runs. These technical measurements
do not determine the generated-quality conclusion. The exact S1 reference remained
unchanged. Selected adapters: `checkpoints/ministral-v2-csft-v1/epoch-02` and
`checkpoints/ministral-v2-dpo-v1/epoch-02`.

The full suite ran 75 tests with one environment-specific native-tokenizer skip.
Native encoding of every prepared prompt/completion succeeded with exact prompt
prefixes, EOS and no truncation. Real GPU smoke tests passed at 2,048 tokens with
exact resumed updates: SFT peak reservation 11.41 GiB; DPO 12.45 GiB. The DPO
synthetic stress pair required three initial FP16 scale reductions, as in D1.
Ruff and launcher syntax checks passed. Synthetic tests verify split isolation,
paired language coverage, current-state precedence and chosen-only SFT dispatch.

Training is visible in the terminal and logged in `reports/logs/`. Commands:

```powershell
.\train_bilingual_pair.ps1
# This fixed launcher refuses to overwrite its completed run directories.
# For a separately named replication:
.\train_bilingual_control.ps1 -Arm sft -RunName ministral-v2-csft-replication
.\train_bilingual_control.ps1 -Arm dpo -RunName ministral-v2-dpo-replication
# Explicitly try the continued-SFT candidate without changing the default:
.\ministral.ps1 -Mode Chat -Profile P1 -Language fr -Adapter .\checkpoints\ministral-v2-csft-v1\epoch-02
```

Data preparation/mining/curation is in `posttraining/preferences_v2.py`; training
in `posttraining/control_train.py`; frozen evaluation in
`posttraining/compare_control.py`; and review aggregation in
`posttraining/summarize_control.py`. Complete checkpoint directories, source
snapshots, fingerprints and curation provenance are preserved. No model default
or advanced method is automatically changed by this experiment.
