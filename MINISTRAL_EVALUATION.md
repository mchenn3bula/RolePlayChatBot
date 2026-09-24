# Ministral P0: local inference and bilingual evaluation

Completed September 24, 2026. **Inference works on the RX 7900 XTX. The initial
bilingual quality gate fails, mainly because of French factual/scene errors.**
No adapter training, memory augmentation, or test-set evaluation was performed.

## Measured runtime

| Item | Result |
| --- | --- |
| Model | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| Exact revision | `b6d637bef2393152b3da2b2fde72eecdee30557e` |
| Loaded parameters | 3,849,090,048, including vision components |
| Runtime | Ubuntu 24.04 WSL2, AMD PyTorch 2.9.1 / ROCm 7.2.1 |
| Separate environment | `/home/n3bula/.venvs/roleplay-ministral` |
| Inference | Frozen weights, FP16, SDPA, batch 1, native Mistral Common tokenizer/template |
| Packages | Transformers 5.17.0, Accelerate 1.15.0, mistral-common 1.12.0 |
| Model load | 5.2 seconds in measured evaluation |
| Panel wall time | 202.8 seconds, including load, preflight, hashing, and saves |
| Timed generation | 189.1 seconds; 6,660 tokens including EOS |
| Aggregate throughput | **35.22 tokens/second**, including per-reply prefill |
| Peak PyTorch reserved | **9.17 GiB** |
| Peak PyTorch allocated | 9.05 GiB, including loading temporaries |
| Final allocated | 7.20 GiB |
| Lowest sampled device-wide free memory | 12.74 GiB |

The source checkpoint is BF16; weights were loaded in FP16 for the established
AMD path. The full multimodal model was loaded, but the panel contains only text.
No offloading or quantization was used. A pinned dependency lock is saved in
`requirements-ministral-rocm.lock.txt`. The working custom-decoder environment
was not modified.

These are short-context inference measurements. Configured context budget is
2,048 tokens; the longest actual input was 585 tokens. They do not establish
2,048-token training, adapter-training memory, or six-hour stability.

## Protocol

The panel was authored before generation in `posttraining/eval_scenarios_v1.json`:
20 development scenarios, consisting of 10 paired English/French scenario
families. Sixteen scenarios have one turn; four have five turns. Seeds 42/43/44
produce **108 replies: 54 per language**, with 60 first-turn replies and 48
follow-ups. Multi-turn histories include the model's actual prior replies, so
earlier mistakes can propagate. Histories reset between scenarios and seeds.

P0 supplies a persona and recent conversation through the native chat template.
It has no separate state ledger, retrieval, response rewriting, or tools.
Generation is fixed at temperature 0.7, top-p 0.9, top-k 50, repetition penalty
1.0, maximum 192 new tokens. Each turn's sampling seed is recorded. Inputs over
the context budget are rejected rather than silently truncated. Expected-answer
annotations are never passed to the model.

Scenarios cover an inn opening, misplaced object, misleading premise, unknown
letter contents, user agency, concise voice, relationships, an invented baker,
parcel handovers, and changing library reservations. All are authored fiction.
No Bluemoon validation/test text or final research test scenarios were consumed.

## Results

All 108 replies were nonempty and ended on EOS; none hit the token limit. Mean
output length was 60.7 tokens excluding EOS. No reply repeated a word 4-gram;
the longest identical-token run was two. This does not imply varied style:
40/108 replies began with an Ah/Oh opening under a simple prefix count.

Every reply was reviewed by the Codex assistant using the saved rubric, expected
behavior, and generated history. **These are non-blinded AI judgments, not
independent human scores.** The human review CSV remains blank. French-language
judgments have not been checked by an independent fluent human reviewer.

| AI review, all turns | English (54) | French (54) |
| --- | ---: | ---: |
| Grammar, mean / 5 | 4.94 | 4.04 |
| Relevance, mean / 5 | 3.78 | 3.17 |
| Coherence, mean / 5 | 3.19 | 2.48 |
| Both relevance and coherence >=3 | 40/54 (74.1%) | 26/54 (48.1%) |
| Replies flagged for explicit factual violation | 14/54 | 22/54 |
| Replies flagged for user-agency violation | 8/54 | 3/54 |
| Requested language used | 54/54 | 54/54 |
| Substantially unintelligible / looping replies | 0 | 0 |

Factual flags count replies with a reviewed violation, not violated fact
opportunities. The research plan's finer opportunity-level endpoint has not yet
been implemented. Omitting a needed correction lowers relevance without
automatically counting as an explicit contradiction. Compatible fictional
embellishments are allowed, although excessive or illogical ones lower coherence.

The predeclared first-turn gate requires >=90% grammar >=3 and >=80% with both
relevance/coherence >=3. Grammar passes at 58/60 (96.7%). Combined relevance and
coherence reach **39/60 (65.0%), so the bilingual gate fails**. By language the
same quality threshold is met in 26/30 English replies (86.7%) and 13/30 French
replies (43.3%). Follow-ups are acceptable in 14/24 English and 13/24 French
replies. These are descriptive pilot counts, not population estimates: only
10 scenario families, paired translations, repeated seeds, and dependent turns.

## Actual examples

English, `en02-s43-t1`: the key is explicitly in the blue drawer, but the user
reaches for the red drawer. The response correctly says:

> The brass key is safely tucked in the blue drawer, where I last checked before you arrived.

French, `fr02-s43-t1`, the corresponding successful correction:

> Ah, non, mon ami ! La clé en laiton est toujours dans le tiroir bleu, là où elle doit rester pour nos prochains ajustements de pendules. Les vis, elles, sont bien dans le rouge, comme je l’ai précisé.

French, `fr04-s42-t1`: the persona explicitly does not know the sealed letter's
contents. The model nevertheless invents a quotation:

> La première phrase, que je peux lire en partie à travers le scelloir, commence par : *« En ce jour où la forêt murmure ses secrets à nouveau… »*

English, `en10-s43-t3`: the user extends a reservation until **tomorrow morning**.
The model changes it to:

> Understood—Jun’s reservation now extends until **sunset tomorrow**.

The wrong deadline persists in that trajectory's final reply. Other failures
include calling a brother a sister, confusing the user with a third-party
customer, inventing parcel contents, and claiming a handover already happened.
The report gallery includes every response, including unsuccessful seeds.

These examples demonstrate useful inference alongside material weaknesses.
They are not a matched improvement claim over the custom model: prompts,
tokenizer, initialization, decoding, and scoring sample differ.

## Runtime checks and limits of diagnosis

The pinned model loaded with finite weights, and preflight logits were finite.
The technical warmup returned a Spanish greeting to an English greeting request;
it was not counted as a semantic pass or included in the fixed roleplay panel.
This reinforces the distinction between numerical health and instruction following.

For three actual failure prompts (`en03`, `fr03`, `fr04`, seed 42), compared the
next-token distribution on identical inputs:

- FP16 default versus forced math attention: maximum absolute logit difference
  <=0.0167; the ordered top five tokens matched in all three.
- FP16 GPU versus FP32 CPU math: maximum absolute logit difference <=0.0169;
  the ordered top five tokens matched in all three. Probability L1 differences
  were <=0.0054.
- BF16 GPU forward was finite on all three; the most likely token matched FP16
  throughout. This only validates those BF16 forwards, not BF16 training.

The source weights are BF16. The diagnostic's CPU cast followed FP16 -> BF16 ->
FP32; tiny conversion effects are possible. These are first-token checks, not
complete cross-device generation equivalence. They give no evidence of a large
ROCm-specific numerical fault on those inputs, but do not certify every kernel,
loader behavior, or identify the cause of all quality failures.

The initial download omitted duplicate consolidated weights intentionally. The
first offline lookup expected every repository file and failed before generation;
matching the download filter fixed it. That failed run remains as evidence in
`reports/ministral-p0-v1-20260924/`. The completed run is the `-r2` directory.

The full offline suite passed **47 tests** in the established ROCm environment.
The six new harness tests also passed in the separate Ministral environment.
Ruff, shell syntax, and PowerShell parsing were checked. The interactive French
chat path was smoke-tested separately from the panel.

## Use it from PowerShell

From the project directory:

```powershell
# Local interactive roleplay; default persona is Mira the innkeeper.
.\ministral.ps1 -Mode Chat -Language en
.\ministral.ps1 -Mode Chat -Language fr
# /reset clears the scene history; /quit exits.

# Repeat the fixed panel in a NEW directory; progress appears in the terminal.
.\ministral.ps1 -Mode Evaluate -RunName ministral-p0-my-repeat
```

For a fresh WSL environment, `bash setup_ministral_env.sh` installs the locked
dependencies. Download the pinned model with:

```bash
$HOME/.venvs/roleplay-ministral/bin/python -m posttraining.download
```

The launchers then use local-only weights. `MINISTRAL_VENV` can override the
environment path. Model cache lives under the WSL user's Hugging Face cache,
outside Git; no model weights or private account configuration are committed.
The default chat retains history up to its budget, then asks for `/reset`.

## Artifacts and next decision

Completed raw evidence: `reports/ministral-p0-v1-20260924-r2/` contains the
manifest, model/source hashes, dependency freeze, frozen panel, exact token IDs,
108 replies, timing/memory, AI annotations, blank human review, runtime comparison,
and `replies.md`. Logs are in `reports/logs/`.
The compact publishable result is `results/ministral-p0-v1.json`.

The run's source snapshot preserves the exact evaluated harness. Subsequent
changes save source snapshots automatically and silence a redundant max-length
warning; they do not change the selected generation length.

**Next:** investigate the failed quality gate with a small, separately named
prompt/state control, especially French false premises and unknown facts. Compare
P0 with correct authored state (P1) before attempting automatic memory or SFT.
If simple explicit state still fails, reassess prompt formatting/backbone
suitability. Keep this completed P0 record intact; do not train adapters merely
because the model fits in VRAM.
