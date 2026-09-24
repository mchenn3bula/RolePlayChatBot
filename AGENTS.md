# RolePlayChatBot project instructions

Read `CODEX_HANDOFF.md` for current state, `BASELINE.md` for the fixed historical
baseline, and `MODERN_BASELINE.md` for the completed modern experiment.
User instructions take precedence over this document.

- Include references for research-backed recommendations and method changes.
  Link primary papers or official documentation near the claims they support;
  distinguish published findings, our hypotheses, and local measurements. Keep
  the README research-reference index current. Preserve frozen experiment files.

- The user authorized continued refinement of natural-reference replies. The
  response-rule experiment in `RESPONSE_REFINEMENT_PROTOCOL.md` adds fresh authored
  bilingual validation, preserves historical ratings and scores third-person
  self-description separately. Generated evaluation review is allowed; private
  source dialogue stays unread. No automatic default promotion or training.
- The user additionally authorized literature research followed by a natural-
  sentence character-reference ablation. See `NATURAL_REFERENCE_RESEARCH.md` and
  frozen `NATURAL_REFERENCE_PROTOCOL.md`. Preserve historical per-text score
  anchors, unchanged negative controls and all previous experiments. This is
  inference-only with the same CSFT checkpoint, not authorization to promote a
  default automatically. Generated evaluation review is allowed in this scope.
- The user authorized a controlled role-confusion/state-format experiment.
  `STATE_FORMAT_PROTOCOL.md` freezes a prose versus entity-assignment rendering
  ablation using the unchanged continued-SFT epoch-02 adapter. Generated replies
  and authored fixtures may be reviewed within this scope; private source dialogue
  stays unread. Preserve its frozen inputs, mappings, criteria and annotations.
  This reuses a familiar development panel, not a new held-out test. No automatic
  model/prompt default promotion or extra training follows from the pilot.
- Latest user override (DPO comparison): the user reviewed the generated dialogue,
  found it appropriate, and permits reading generated replies when needed to
  compare standard DPO with SFT. Limit this exception to generated evaluation
  replies and authored synthetic preference fixtures; raw source conversations
  remain private. Clearly distinguish assistant judgments from human ratings.
  Logs should still omit dialogue. Preserve historical reports and checkpoints.
- Standard DPO D1 is now trained from S1 epoch-02, also its immutable reference.
  See `DPO_COMPARISON.md` and the frozen comparison protocol. The 160/40 preference
  rows are assistant-authored synthetic contrasts from 16/4 template families,
  not human labels. Selected checkpoint: `ministral-d1-standard-v1/epoch-02`.
  Keep this pilot separate; no automatic promotion of the chat default.
- The user authorized strengthening bilingual preferences and a continued-SFT
  control. See `BILINGUAL_CONTROL_PROTOCOL.md` and `BILINGUAL_CONTROL_RESULTS.md`.
  V2 uses 40 training scene families, one EN/FR pair per scene, and separately
  held-out validation/evaluation scenes. Curation may inspect new authored fixtures
  and their generated candidates; private source conversations remain unread.
  Labels are assistant-curated unless the user supplies independent annotations.
  The continued-SFT and standard-DPO arms start from S1, use identical chosen
  exposure/updates, and are not equal-compute claims. Advanced methods stay deferred.
- Earlier user output-review preference (superseded only within that exception): do not read, quote, or judge
  roleplay dialogue or model-generated replies. Do not run content-scoring or
  AI-review pipelines. Save replies for the user and provide file locations.
  Verify changes with authored benign fixtures and technical checks only.
  Keep future generation logs content-free; do not open generated galleries,
  transcripts, or raw generations in tools. Historical reviews are superseded
  as a workflow by this preference. The local chat model may process history
  to run the application, but the coding assistant must not inspect replies.

- Preserve the architecture-version-2 dense baseline: four layers, width 256,
  four heads, feed-forward width 1,024, maximum length 1,024, tied embeddings.
  Treat architecture/data/objective changes as explicitly named experiments.
- The original compressed-attention model leaked future tokens. Historical
  perplexity near 6.3 is not a valid result for the corrected model.
- Preserve reply-only next-token loss, exact EOS boundaries, padding isolation,
  tokenizer fingerprints, and thread-level train/validation/test separation.
- Use validation for iteration. Reserve test data for final evaluation. Report
  generated-text quality alongside perplexity; compare different tokenizers using
  shared prompts and quality judgments rather than raw token perplexity.
- Keep complete checkpoint directories together. Never overwrite a completed run
  or silently relax resume compatibility checks. The completed Colab checkpoint is
  on the user's Drive, not in this transfer archive.
- The RX 7900 XTX is validated for FP16 baseline training under Ubuntu 24.04
  WSL2 with ROCm 7.2.1 / AMD PyTorch 2.9.1; see `AMD_TRAINING.md`. All 28 tests,
  real-data smoke train/resume/reload, and a short benchmark passed. Full FP16
  training later completed for the modern experiment; BF16 remains untested.
  Do not install `requirements-cuda.txt` on
  AMD or change model architecture to make a runtime environment work.
- The user's six-hour overnight experiment is `overnight-124m-v1`, configured
  in `configs/overnight_124m.json`: 12 layers, width 768, 12 heads, FF 3,072,
  1,024-token maximum length, 124,337,664 parameters. See `OVERNIGHT_BASELINE.md`.
  `train_amd.ps1 -Profile Overnight` selects four planned epochs with a six-hour
  resumable soft limit, FP16, micro-batch 8 / accumulation 4, and no gradient
  checkpointing. Real trainer smoke/resume/reload passed; peak reservation was
  15.28 GiB. Benchmark throughput was 36.28 examples/sec. Full training has not
  started. Preserve this as a separate experiment from the historical baseline.
- Ministral P0 inference is established in a separate WSL environment at
  `$HOME/.venvs/roleplay-ministral`; pinned model weights are in the HF cache.
  The completed P0 run saved 108 EN/FR replies. Its historical report contains
  generated text: do not open it under the user's current output-review preference.
  The user has authorized a whole-conversation LoRA S1 pilot. See
  `LORA_TRAINING.md`; selected dialogue files are user-only and must not be opened
  by the coding assistant. Structural curation is not semantic quality review.
  S1 finished two epochs / 30 updates; selected adapter is
  `checkpoints/ministral-s1-whole-lora-v1-r2/epoch-02/`. Held-out reply NLL went
  from 2.88195 to 2.68817; this is not a generated-quality judgment. Peak training
  reservation was 11.41 GiB, with no FP16 overflow retries.
  Use `train_lora.ps1` and the separate `roleplay-lora` environment. Do not alter
  running trainer/config/runtime source files, whose hashes guard resume.
  Preserve P0 and P1 as separate controls. P1 authored state is implemented; see
  `PERSONA_SCENE_STATE.md`. The suite has 63 tests; the native tokenizer test is
  additionally run in the LoRA environment when unavailable in the baseline env.
  New inference
  runs save file-only replies and technical summaries, with no AI/content scoring.
  P1 is the default launcher profile; preserve P0 as a separately selectable control.
  Use the user's own output judgments to decide later training work.
  Follow the September 23 `RESEARCH_PLAN.md` and `LITERATURE_REVIEW.md`: first
  explicit-state controls after P0 inference, then curated SFT and DPO.
  Se-DPO/distillation/RL are gated alternatives, not mandatory stacked stages.
  Use a separate environment and measured AMD preflight before a six-hour run.
- The user adopted RoPE + SwiGLU and reply-position-only vocabulary projection
  as `modern-rope-swiglu-v1`, architecture version 3, 123,551,232 parameters.
  Use `configs/overnight_rope_swiglu.json` and `train_amd.ps1 -Profile Modern`.
  This keeps 12 layers / width 768 / 12 heads / length 1,024; SwiGLU width is
  2,048 and RoPE theta is 10,000. Four planned epochs and a six-hour resumable
  soft cap remain. All 41 tests and the actual-model GPU smoke/resume/reload
  passed. The full run completed four epochs with best validation perplexity
  38.10. The fixed generation panel is complete; see `GENERATION_EVALUATION.md`.
  Replies have weak scene/character tracking; the historical short inn prompt
  produces gibberish. Do not treat the low word-repetition metric as proof of
  useful replies or claim a matched improvement over the unavailable T4 model.
  New launcher runs show terminal progress
  and save unique logs in `reports/logs/`. Preserve v2 checkpoint behavior.
- After behavioral model/training changes, run `python -m unittest discover -s
  tests -v` in the configured environment. Tests use synthetic data and do not
  require downloading a model. Check the affected files with Ruff if installed.
- `build_colab_bundle.py` builds the separate Colab upload archive.
  `build_transfer_bundle.py` builds the full personal transfer archive.
- Keep `.venv`, secrets, account configuration, data, generated archives, and
  checkpoints out of Git. The transfer archive includes prepared data for the
  user's own continuation; it is not a public-release dataset package.

Project instructions follow the documented
[AGENTS.md convention](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
