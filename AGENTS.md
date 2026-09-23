# RolePlayChatBot project instructions

Read `CODEX_HANDOFF.md` for current state, `BASELINE.md` for the fixed historical
baseline, and `MODERN_BASELINE.md` for the user's selected next experiment.
User instructions take precedence over this document.

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
- Mistral/Ministral adapter training and the research plan are proposed work;
  no pretrained backbone or adapters have been downloaded or trained here.
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
