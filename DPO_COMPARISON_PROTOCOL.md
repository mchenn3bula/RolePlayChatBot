# Prespecified SFT versus standard DPO pilot

Frozen before DPO training and comparison output review, September 24, 2026.
User authorization permits generated-reply review for this comparison. Private
source conversations are not read. New preference examples are authored synthetic
fictional scenes, not human-labeled preferences.

## Arms and training

- SFT: immutable `checkpoints/ministral-s1-whole-lora-v1-r2/epoch-02`.
- DPO: initialize the exact same adapter from SFT; rank 16, attention q/k/v/o only.
- Fixed reference: exact SFT adapter, cached summed completion log probabilities
  with the native tokenizer and EOS. Disabling the adapter is **not** the reference.
- Original sigmoid DPO: `-log sigmoid(beta * ((log pi chosen - log pi rejected)
  - (log ref chosen - log ref rejected)))`. Equal pair weight, no length
  normalization, no SFT auxiliary term, no label smoothing, no token weighting,
  no reference updates, and no reward-model or online-RL stage.
- Beta 0.1, peak LR 5e-6, 10% warm-up then cosine decay, micro-batch 1 pair,
  accumulation 8, two epochs, one training seed 20260925, dropout disabled.
- 160 train / 40 validation pairs, respectively 16 / 4 distinct authored scene
  templates. Five aliases and both languages of each template stay in one split.
  The 200 rows are correlated augmentations, not 200 independent scene families.
- Select the trained epoch with lowest synthetic held-out DPO loss. One fixed
  beta/LR recipe; no tuning on generated comparison replies. Maximum six hours,
  18 GiB reserved / 4 GiB device-free guards. Preserve all earlier checkpoints.

The objective follows the [original DPO paper](https://arxiv.org/abs/2305.18290).
Fixed-reference caching and disabled dropout are supported in the
[official DPO trainer documentation](https://huggingface.co/docs/trl/dpo_trainer).
The local implementation is independently tested for shifted reply masks, EOS,
sequence sums, pair averaging, analytic gradients and exact checkpoint resume.

## Identical evaluation inputs

Freeze the exact 108 message arrays, state snapshots, IDs and per-turn sampling
seeds from the completed SFT development panel before loading either comparison
arm. Generate fresh SFT and fresh DPO responses from those immutable arrays.
Each arm must reproduce every original input token ID exactly. Compare input
hashes, config, model revision, tokenizer and adapter hashes in the result manifest.

Both arms use the same P1 state wrapper, 2,048-token input budget, maximum 192 new
tokens, temperature 0.7, top-p 0.9, top-k 50, repetition penalty 1.0, FP16, SDPA,
native Mistral template and seeds 42/43/44 with the existing per-turn seed formula.
Do not use model-specific decoding or postprocessing. Reserve test data.

The 60 first-turn requests have no generated history. The 48 follow-up requests
share **historical SFT-generated history**, which is an asymmetric source of
context, but the exact same input is supplied to both arms. This is a controlled
response comparison, not an independent closed-loop conversation comparison.
Report first turns and follow-ups separately. No evaluated prompt or completion
is used for DPO training or checkpoint selection.

## Review rubric frozen before outputs

Randomize A/B order independently per request; hide the model mapping during the
assistant's review. Report the review as a **single, non-independent AI reviewer**
who authored the preference templates, not a blinded human study. Human review
remains valuable. User approval of appropriate content is not a preference label.

Primary descriptive comparisons:

1. Paired win/tie/loss on relevance, consistency with explicit facts, user agency,
   language and fluency. Prefer satisfying the request over verbosity; choose a
   tie when differences do not materially improve the response.
2. Violated required fact opportunities / required opportunities, with omissions
   reported separately. An unanswered or unreadable reply is not rewarded for
   avoiding contradictions. Use the fixed slot map in the comparison runner.

Secondary: requested-language adherence, coherence and relevance (1-5), grammar
(1-5), user-agency violations, empty outputs, within-response repeated word
4-grams, EOS/cap counts, token counts, timing and memory. Token/length metrics
are descriptive, not quality proxies. Compatible scene elaboration is permitted;
inventing something explicitly unknown is a fact violation. Report examples of
both improvements and regressions, including ties and negative outcomes.

Group uncertainty estimates by the ten scenario families, keeping their language
translations, seeds and turns together. A 10,000-resample cluster bootstrap is
descriptive only: ten families, one training seed and a familiar development
panel do not support a broad claim of method superiority. The comparison measures
SFT versus SFT plus extra synthetic preference training, not the isolated effect
of changing an optimizer under identical training data/compute.
