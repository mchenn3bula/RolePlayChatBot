# Generation evaluation — September 23, 2026

**Result: the trained model can imitate roleplay prose, but it is not yet a
reliably coherent or relevant chatbot.** Severe word loops were uncommon in
the prepared validation panel, but generic phrasing, character changes, and
scene drift remain. The original short inn-door prompt failed badly.

## Completed training and evaluation protocol

Run: `modern-rope-swiglu-v1`, 123,551,232 parameters, architecture v3 with
RoPE and SwiGLU. Four epochs completed with 22,210 successful optimizer updates
and six FP16 overflow skips. Validation perplexity by epoch: 52.96, 43.03, 38.68,
**38.10**. `best.pt` is the fourth-epoch checkpoint, SHA256
`c8e2a615586d188b4d0979ccfdd9d39702d24e9165043b6e2593095e2a7bd831`.

The fixed panel used 50 validation contexts sampled with selection seed 42 and
three generation seeds, 42/43/44: **150 replies**. Settings were unchanged from
`BASELINE.md`: FP32 inference, temperature 0.8, top-k 50, top-p 0.9, repetition
penalty 1.05, maximum 128 new tokens. Exact prepared context IDs and their EOS
boundary were used; no test-split examples were read. The manifest records
checkpoint, dataset, configuration, source hashes, prompts, and sampling settings.

## Automatic results across all 150 replies

| Measure | Result |
| --- | ---: |
| Empty replies | 0 / 150 |
| Replies with fewer than four words | 0 / 150 |
| Mean generated length, excluding EOS | 77.3 tokens |
| Median generated length | 73.5 tokens |
| Stopped on EOS | 104 / 150 (69.3%) |
| Hit the 128-token cap | 46 / 150 (30.7%) |
| Replies containing any repeated word 4-gram | 30 / 150 (20.0%) |
| Mean within-reply repeated word 4-gram fraction | 0.41% |
| Highest within-reply repeated word 4-gram fraction | 8.60% |

The 0.41% measure is the mean fraction of duplicate four-word windows within
each reply, not the percentage of repetitive replies. It misses repeated short
phrases, semantic redundancy, and subword loops. The most repetitive panel
reply repeatedly uses “he admitted” and duplicates “I'm pretty sure.” Many
other outputs lean on smiling, nodding, sighing, and vague agreement. The seeds
are repeated samples of the same 50 prompts, not 150 independent situations.

## Detailed qualitative review

The Codex assistant reviewed the full contexts and seed-42 generations for the
first **20 prompts in the fixed selection order**. This selection was stated
before completing the panel and did not choose the best or worst generation.
These are subjective AI ratings, not independent human annotations. The
original `human_review.csv` is left blank for a human reviewer.

| Dimension | Mean score out of 5 |
| --- | ---: |
| Grammar/readability | 3.0 |
| Relevance to the immediate scene | 2.1 |
| Coherence and continuity | 2.0 |

Only **3 of 20** reviewed replies scored at least 3/5 on both relevance and
coherence. Scores and specific reasons for every reviewed example are saved in
`assistant_review.json`, alongside the rubric. This small sample describes
observed failures; it is not a calibrated population-wide quality estimate.

Representative observations, all seed 42:

- **Validation index 24299:** a short exchange between spouses gets a plausible
  response from the correct character. “Megohime made a noise, but nodded” is
  locally consistent. The reply is simple and introduces an unmotivated movement.
- **Index 8024:** a snowbound indoor scene ends with a mysterious knock. The model
  moves Sasuke into a forest and writes “he could hear a soft frown on his face.”
  Both scene continuity and sentence meaning fail.
- **Index 2848:** the prompt discusses a recovering Vulpix and an invitation to
  spend the day together. The answer instead introduces Dr. Moreau and a lab.
- **Index 7314:** the reply partly answers a request to sleep together, but replaces
  Rosalina with Kori and loses the established location.
- **Index 1041:** the output keeps the baby/family topic while making mutually
  inconsistent statements about having, finding, and collecting a child.
- **Index 7164:** the model retains Diego and a battle but introduces an unsupported
  character and produces unclear physical events rather than following the action.

The recurring pattern is surface style imitation with weak tracking of who is
speaking, what just happened, and what a reply needs to address.

## Separate short-prompt diagnostic

The historical prompt was also tested, separately from the validation panel:

> The traveler knocks on the inn door. "Is anyone there?"

All three seeds produced garbled, irrelevant text. Seeds 42 and 43 ended on EOS;
seed 44 reached the 128-token cap and contained **19 consecutive copies of one
token**. The word 4-gram metric misleadingly reports zero repetition for these
strings because many repeated subwords merge into long nonsensical words.

Follow-up checks confirmed an exact prompt/tokenizer round-trip, the correct
EOS boundary, and the same seed-42 output through the actual `generate_chat`
API. CPU and GPU first-token logits agreed (maximum absolute difference about
0.000031), as did the default and explicit math GPU attention backends. FP16
and FP32 both put similarly unrelated tokens at the top of the first-token
distribution. These checks make a tokenizer mismatch or the checked GPU
attention-path discrepancy unlikely explanations. They do not establish the
training/data root cause or show that every short prompt fails.

## What can be claimed

Validation perplexity improved substantially from the historical 16.3M model's
reported 75.68. The new model often produces recognizable roleplay sentences
and avoids a persistent word loop on the prepared validation contexts. However,
**we cannot establish a matched reduction in repetition versus the old model**:
its trained checkpoint and fixed generation panel are not available locally.
The old anecdotal door repetition has not been resolved into a useful reply;
the new model produces gibberish on that prompt instead.

Treat this as a completed, evaluated from-scratch baseline with substantial
quality limitations. Before another long run, investigate the short-prompt
failure and context/character tracking using a separately recorded diagnostic
set. Do not conclude that more epochs, a larger model, or higher repetition
penalties alone will fix it. Keep the test split reserved.

## Saved evidence

The paths below identify local artifacts, which are excluded from Git along with
datasets and checkpoints. A compact public record of the training metrics,
aggregate generation metrics, review ratings, and evaluation configuration is
available in `results/modern-rope-swiglu-v1.json`; it contains no raw conversations.

- `checkpoints/modern-rope-swiglu-v1/metrics.json`
- `reports/modern-rope-swiglu-v1-validation-20260923/manifest.json`
- `reports/modern-rope-swiglu-v1-validation-20260923/generations.jsonl`
- `reports/modern-rope-swiglu-v1-validation-20260923/summary.json`
- `reports/modern-rope-swiglu-v1-validation-20260923/additional_metrics.json`
- `reports/modern-rope-swiglu-v1-validation-20260923/assistant_review.json`
- `reports/modern-rope-swiglu-v1-validation-20260923/human_review.csv`
- `reports/modern-rope-swiglu-v1-inn-20260923.json`
- `reports/modern-rope-swiglu-v1-inn-probe-20260923.json`
- `reports/logs/modern-generation-eval-20260923.log`

No weights, training settings, decoding defaults, or prepared datasets were
changed during this evaluation.
