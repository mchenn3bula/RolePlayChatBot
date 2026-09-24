# Frozen bilingual preference and continued-SFT control protocol

Fixed before mining, training or viewing new evaluation outputs, September 24, 2026.
Scope: ordinary continued SFT and standard DPO only. No advanced objective or RL.
Private source dialogue remains unread. New fictional fixtures and generated
comparison/curation replies may be inspected under the user's request.

Data: retain one alias per D1 template (16 train / 4 validation scenes); add
24 training, 8 validation and 8 evaluation scenes, each in English and French.
Totals: 80 training pairs from 40 scene families, 24 validation pairs from 12,
and 16 evaluation prompts from 8. Languages of a scene remain in the same split.
The new scenes vary facts, roles and actions, rather than just character names;
they still share task patterns. They are not 40 independent reasoning mechanisms.
Some requests include an authored earlier exchange with superseded facts. Current
P1 state controls the answer. Historical development examples are not used for
training. The new evaluation is a development holdout, not the locked final test.

Generate one reply from immutable S1 for each new train/validation prompt (64).
Inspect every candidate: use it as rejected only if a clear error makes the
authored response preferable. Otherwise keep the authored contrast. Record a
reason and provenance for every decision; fluent correct alternatives are not
negative labels. Keep exactly one pair per scene/language to avoid overweighting
failure-heavy scenes. Labels are assistant judgments unless separately supplied
by the user; no independent human review is presumed. Freeze the final dataset
and its fingerprints before training. Candidate generation never uses evaluation
prompts. Provide a bilingual review file for the user.

Three comparison arms: unchanged S1; continued SFT from S1 on the chosen responses;
standard DPO from S1 using the same pairs and exact frozen S1 reference. Both
trained arms use rank-16 q/k/v/o LoRA, FP32 adapters/FP16 backbone, dropout off,
LR 5e-6, beta 0.1 for DPO, accumulation 8, two epochs, same shuffled order and
training seed 20260928. The primary checkpoint is the final second epoch in both
arms: equal 20 optimizer updates and 160 chosen-response presentations. No
quality-based selection or hyperparameter adjustment after viewing outputs.
SFT uses conventional target-token-weighted reply CE; DPO uses equal-pair sigmoid
loss with summed completion log probabilities and native EOS.

This is **exposure/update matched**, not equal-FLOPs or wall-time matched: DPO also
processes rejected replies and a reference cache. Record actual chosen/rejected
target tokens, complete sequence tokens, training seconds and peak memory; do
not claim equal compute. A separate exact compute-budget ablation is outside this
pilot. Holding positive data, initialization, modules, updates and LR fixed makes
the simpler extra-supervision explanation assessable, but does not optimize each
method's hyperparameters independently.

Evaluate all three arms on the same 16 new prompts at seeds 42/43/44 (48 responses
each), identical native token IDs, authored history and P1 state, maximum 192 new
tokens, temperature 0.7, top-p 0.9, top-k 50, repetition penalty 1.0, FP16/SDPA.
No generated histories or arm-specific postprocessing. Randomize A/B/C labels per
request and hide mapping until assistant annotations are frozen. Judge all outputs
against the authored scene, not exact matching with the canonical chosen wording.

Primary: all required facts and role/agency constraints satisfied, per response,
plus contradiction and omission flags separately, by language. Secondary:
coherence/relevance/grammar 1–5, language compliance, agency violations, empty/capped
replies, output length, repetition and generation cost. Record pairwise preference
between continued SFT and DPO (tie allowed); S1 is a baseline diagnostic. Bootstrap
that preference by the eight scenario families, 10,000 resamples; uncertainty is
descriptive given one seed and a single non-independent assistant reviewer.
Report unchanged, worse and improved examples. Do not promote on aggregate gains
that hide a French regression. No automatic model-default change or advanced run.

Use six-hour resumable caps per training arm, visible terminal/file progress,
18 GiB reserved and 4 GiB free-memory guards. Preserve all completed checkpoints
and the original D1 protocol/review. Verify masks, split isolation, initialization,
SFT chosen-only training, exact resume and paired input equality with synthetic
tests and GPU smoke before actual training. Run the full configured test suite.
