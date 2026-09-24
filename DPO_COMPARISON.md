# Standard DPO versus SFT: completed pilot

September 24, 2026. **DPO gives a modest improvement in this development panel,
but does not reduce the total number of required-fact contradictions. Keep SFT
as the comparison baseline; do not promote DPO as a reliable bilingual upgrade.**

## Matched reply comparison

The single assistant reviewer preferred DPO on 27 of 108 paired requests, preferred
SFT on 14, and judged 67 ties. Labels were randomized A/B; annotations were frozen
and hashed before opening the model mapping. The reviewer also authored the
synthetic preference curriculum, so this is **not an independent or human study**.

| Panel | DPO wins | Ties | SFT wins |
|---|---:|---:|---:|
| All 108 | 27 | 67 | 14 |
| English, 54 | 15 | 33 | 6 |
| French, 54 | 12 | 34 | 8 |
| First turns, 60 | 21 | 27 | 12 |
| Shared-history follow-ups, 48 | 6 | 40 | 2 |

| Measurement | SFT | SFT + standard DPO |
|---|---:|---:|
| Required-fact contradictions / 174 opportunities | 27 (15.5%) | 27 (15.5%) |
| Required-fact omissions / 174 | 61 (35.1%) | 49 (28.2%) |
| English contradictions / 87 | 9 | 6 |
| French contradictions / 87 | 18 | 21 |
| Coherence, assistant rating / 5 | 3.56 | 3.81 |
| Relevance, assistant rating / 5 | 3.65 | 3.92 |
| Grammar, assistant rating / 5 | 4.82 | 4.88 |
| User-agency violations / 108 | 6 | 6 |
| Requested language followed / 108 | 108 | 108 |
| Mean repeated word 4-gram fraction | 0.248% | 0.094% |
| Mean output tokens, including EOS | 21.90 | 21.31 |
| Original SFT validation reply NLL | 2.688166 | 2.687299 |

Fact counts cover the prespecified required slots, not every possible error.
Other problems, such as becoming the mayor instead of the archivist, enter the
quality ratings and review notes. A slot can be contradicted, omitted or supported;
contradictions and omissions do not overlap. Contextually clear references count
as support without requiring every name/adjective to be repeated. Borderline
decisions and pronoun ambiguities are documented in each annotation.

There were 39 text-identical pairs. Neither arm produced an empty response or hit
the 192-token cap; all 108 replies in each arm ended with native EOS. Short replies
partly explain the low repetition figures; those figures do not establish quality.
The retention check uses the same 17 validation conversations / 12,702 target
tokens, processed numerically without decoding private source dialogue. The tiny
NLL difference is not evidence of a meaningful generalization gain.

Counting ties as half a win gives DPO a 56.0% preference share. A descriptive
10,000-resample bootstrap over the ten scenario families gives 50.9–61.7%.
Translations, seeds and turns remain clustered. This interval does not capture
reviewer bias, training-seed variation or generalization to new scenario families.
It must not be presented as broad statistical proof that DPO outperforms SFT.

## Examples, including regressions

- Improvement, `en10-s42-t3`: after an explicit extension until tomorrow morning,
  SFT says “until sunset tomorrow”; DPO says “until tomorrow morning.”
- Improvement, `fr10-s42-t2`: at noon, SFT incorrectly expires a sunset reservation;
  DPO correctly says it has not yet expired.
- Regression, `fr09-s42-t2`: after arriving at the bakery, SFT says
  “Nous sommes dans la boulangerie.” DPO says “Nous sommes en route vers la
  boulangerie,” reverting to an earlier location.
- Regression, `en02-s43-t1`: SFT supplies both the key's blue drawer and the screws'
  red drawer; DPO supplies the key location but omits the screws.
- Unresolved tie, `en09-s42-t3`: both answer “I am carrying the red umbrella,”
  omitting the green parcel they were explicitly handed and asked to keep.

Read all pairs in `reports/ministral-sft-vs-dpo-v1/comparison.html`.
The original SFT run, fresh matched arms, blinded annotations and frozen review
hash remain preserved. No private Bluemoon source conversation was read.

## What was trained

SFT is the immutable `ministral-s1-whole-lora-v1-r2/epoch-02` adapter over pinned
Ministral 3 3B Instruct. DPO initializes that exact adapter and uses it as its fixed
reference, rather than disabling the adapter and accidentally referencing the
original pretrained backbone. Both retain the rank-16 attention q/k/v/o adapter,
9,371,648 FP32 trainable parameters, and frozen FP16 backbone on the RX 7900 XTX.

This is the original sigmoid DPO objective: summed completion log probabilities,
native EOS, prompt masking, equal pair weighting, beta 0.1, peak LR 5e-6, 10% warmup
then cosine decay, accumulation 8, two epochs, one seed, dropout disabled. No SFT
auxiliary loss, length normalization, label smoothing or online reward model.
See [the original DPO paper](https://arxiv.org/abs/2305.18290) and the frozen
[local protocol](DPO_COMPARISON_PROTOCOL.md).

The curriculum has 160 training / 40 validation synthetic preference pairs from
16 / 4 distinct templates. Each template has five aliases and EN/FR versions,
kept together in one split. These are assistant-authored controlled contrasts,
not human-labeled preferences, natural model failures or 200 independent scenes.
No evaluation examples or test data were used to train or select the checkpoint.

Training completed 40 updates in 234.8 seconds with no overflow retries and
9.17 GiB peak reservation. Validation DPO loss fell from 0.6931 to 0.6341 after
epoch 1 and 0.6130 after epoch 2. Epoch 2 was selected by that loss before output
review. Raw chosen-over-rejected sequence-likelihood accuracy stayed 87.5%; the
relative margin improved. The SFT reference checkpoint and cache hashes stayed
unchanged. The selected adapter is `checkpoints/ministral-d1-standard-v1/epoch-02`.

## Identical evaluation conditions and limits

Both arms received byte-identical message arrays and identical native input token
IDs, state snapshots, per-turn sampling seeds and decoding configuration. The base
model revision is `b6d637bef2393152b3da2b2fde72eecdee30557e`. Inputs use the P1 state
wrapper, native Mistral template, FP16/SDPA, 2,048-token input budget, 192 maximum
new tokens, temperature 0.7, top-p 0.9, top-k 50, repetition penalty 1.0 and seeds
42/43/44. No arm-specific postprocessing was used. Fresh SFT reproduced all 108
historical outputs token for token.

The 48 follow-up requests share historical **SFT-generated** context. This controls
the input but favors one source of history; it does not measure independently
evolving DPO conversations. Only the 60 first-turn requests avoid that dependency.
Generation ran sequentially on the same machine, around 19.7 versus 19.6 tokens/sec,
with 9.22 GiB peak reservation in both arms. Timing is descriptive, not a dedicated
controlled-performance benchmark; CPU verification briefly overlapped DPO generation.

This measures SFT against **SFT plus additional synthetic preference training**.
It does not isolate the optimizer under equal training data and compute. A future
objective ablation would need a continued-SFT arm trained on the chosen responses
with a matched budget. No model default was changed by this comparison.

## Next experiment

Keep D1 available for manual comparison, but retain S1 as the control. Prioritize
French role assignment, current object ownership, deadline changes and explicit
false-premise corrections. Collect user-reviewed preferences on varied, natural
model mistakes, including correct but less stylish responses as hard negatives;
avoid only making the rejected answer obviously wrong. Use new bilingual scenario
families reserved before training, then compare SFT, matched continued SFT and
standard DPO. Add independent conversation rollouts and more training seeds before
deciding whether to promote an adapter. Do not stack RL or distillation onto this
pilot merely because preference loss improved.

## Reproduction and verification

`train_dpo.ps1` provides visible progress, unique logs, a six-hour soft cap and
strict resumable checkpoints. Completed runs cannot be overwritten or resumed.
For a new experiment name:

```powershell
.\train_dpo.ps1 -Mode Train -RunName ministral-d1-replication-v1
.\ministral.ps1 -Mode Chat -Profile P1 -Language fr -Adapter .\checkpoints\ministral-d1-standard-v1\epoch-02
```

The matched runner is `python -m posttraining.compare_dpo`: `prepare` freezes the
source requests; `generate --arm sft|dpo --adapter <directory>` runs each arm;
`pair` verifies input equality and writes the randomized review plus labeled gallery.
All modes require `--output-dir`; `prepare` additionally requires `--source-run`.
Use the validated WSL LoRA environment. The frozen annotations are aggregated by
`python -m posttraining.summarize_dpo --output-dir reports/ministral-sft-vs-dpo-v1`.

Validation passed: 71 full-suite tests with one native-tokenizer skip in the
baseline environment; all seven LoRA/native-tokenizer tests passed in the LoRA
environment. The real-model 2,048-token DPO smoke test passed with 12.45 GiB peak
reservation, exact resumed updates and unchanged SFT reference. Its synthetic
stress pair needed three initial FP16 scale reductions; real training needed none.
Ruff, Bash syntax and PowerShell parsing passed. Logs and machine-readable results
are in `reports/ministral-sft-vs-dpo-v1/` and `results/ministral-d1-vs-sft-v1.json`.
