# RolePlayChatBot

## Research references

Paper citations, source links, limitations and their connection to our experiments
are collected in:

- [Literature review](LITERATURE_REVIEW.md): data curation, DPO/Se-DPO, memory,
  roleplay evaluation, distillation and reinforcement learning.
- [Character-reference research](NATURAL_REFERENCE_RESEARCH.md): SPASM, RoleMRC,
  prompt-format sensitivity, semantic evaluation and anonymous roleplay benchmarks.
- [Advanced-method decision](ADVANCED_METHOD_DECISION.md): the Se-DPO reference
  and why the available evidence did not justify an advanced training run.

These sources motivate hypotheses; our measured results are reported separately
in the experiment reports below. Future research-backed changes should include
primary-source references and distinguish published evidence from local findings.
The [training mathematics](#training-mathematics) below defines our reply-only
objective, LoRA parameterization and standard DPO loss, with implementation links.

## Response-rule refinement and fresh scenes

An added first-person/completeness rule improved semantic passes on new authored
EN/FR scenes from 10 to 14/36, but reduced familiar-scene passes from 41 to 39/48.
The acceptance gate failed; the default remains unchanged. Stale possession
updates and incomplete answers remain weaknesses. Read
[RESPONSE_REFINEMENT_RESULTS.md](RESPONSE_REFINEMENT_RESULTS.md); all paired replies
are in `reports/ministral-response-refinement-v1/comparison.html`. These are
single-assistant judgments, not independent human evaluation. No training was run.

## Natural-sentence character references

A paper-informed controlled follow-up reduced observed role errors from 13 to
zero and improved all-constraint passes from 30 to 41/48, but omissions increased
from four to seven. It remains experimental; the default is unchanged. Read
[NATURAL_REFERENCE_RESULTS.md](NATURAL_REFERENCE_RESULTS.md) and the
[research references](NATURAL_REFERENCE_RESEARCH.md). Full paired replies and
inputs: `reports/ministral-natural-reference-v1/comparison.html`. Scores are
non-independent assistant judgments on a familiar development panel.

## Controlled state-format comparison

Explicit entity assignments reduced role confusion but increased omissions in the
fixed continued-SFT model. The default remains unchanged. See
[STATE_FORMAT_RESULTS.md](STATE_FORMAT_RESULTS.md) for paired EN/FR results and
`reports/ministral-state-format-v1/comparison.html` for every reply and exact prompt.
The frozen experiment used no training; ratings are assistant judgments on a
familiar development panel, not independent human evidence.

## Bilingual preferences and continued-SFT control

The v2 comparison adds varied bilingual scenes, assistant-reviewed SFT mistakes
and a continued-SFT arm using the exact chosen replies given to DPO. Read
[BILINGUAL_CONTROL_RESULTS.md](BILINGUAL_CONTROL_RESULTS.md) for outcomes and
[BILINGUAL_CONTROL_PROTOCOL.md](BILINGUAL_CONTROL_PROTOCOL.md) for the frozen
exposure/update-matched design. Independent human preference review remains absent.

## Standard DPO versus SFT

The standard-DPO pilot starts from the completed SFT adapter and compares both
using identical evaluation inputs and decoding. See [DPO_COMPARISON.md](DPO_COMPARISON.md)
for measurements, limitations, and the side-by-side gallery location. The user
permits assistant review of generated replies for this comparison; private source
conversations remain unread. Older user-only review notes below describe earlier
runs and remain the default outside this explicit exception.

## Whole-conversation LoRA

A separate rank-16 Ministral LoRA pilot uses structurally filtered whole source
conversations. See [LORA_TRAINING.md](LORA_TRAINING.md) for data selection,
training/resume commands and explicit adapter loading. Dialogue and generated
replies are saved for user review only; the coding assistant checks technical
metrics without reading their contents.

## Explicit persona and scene state

Ministral chat now defaults to the P1 authored-state profile. Start with
`.\ministral.ps1 -Mode Chat -Profile P1 -Language en` (or `fr`). The terminal
shows an editable session state file and the path to `replies.html`; open that
file in your browser and refresh for replies. Use `/reload` after editing state
and increasing its revision. State updates are explicit, not extracted from chat.

See [PERSONA_SCENE_STATE.md](PERSONA_SCENE_STATE.md) for the schema, commands and
context limits. P0 remains selectable. The coding assistant does not read or
judge replies; new runs log technical progress and save outputs for user review.

The modern four-epoch run and fixed validation generation panel are complete.
See [GENERATION_EVALUATION.md](GENERATION_EVALUATION.md): perplexity improved,
but reply coherence and relevance remain weak, with a severe short-prompt failure.

The [Ministral P0 inference evaluation](MINISTRAL_EVALUATION.md) is complete:
108 English/French replies, 35.2 tokens/sec, 9.17 GiB peak reservation. The model
runs locally, but the bilingual quality gate fails on factual/scene consistency.
Use `./ministral.ps1 -Mode Chat -Language en` (or `fr`) for interactive inference.
The revised [research plan](RESEARCH_PLAN.md) prioritizes explicit-state controls
before curated SFT and matched DPO experiments. See the
[September 2026 literature review](LITERATURE_REVIEW.md) for the research basis.

The completed PC experiment uses **RoPE + SwiGLU, 123.6M parameters**.
See [MODERN_BASELINE.md](MODERN_BASELINE.md) for checked preflight results,
the six-hour recipe, and live terminal/file logging with `-Profile Modern`.

A small decoder-only language-model training project for roleplay conversations.
The corrected model uses **causal self-attention and one next-token prediction
head**. A dense model is the default; a repaired sparse mixture of experts is
available as an optional experiment.

The historical project name was Mini-DeepSeek. Its original attention used
Linformer-style sequence compression, not a faithful reproduction of DeepSeek's
MLA. That implementation has been replaced with standard causal attention;
the archived notebook, PDF, and diagram document the earlier design.

For this PC's RX 7900 XTX training environment and launcher, see
[AMD_TRAINING.md](AMD_TRAINING.md). The code review and baseline tradeoffs are in
[ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md).
The larger 124M six-hour experiment is specified and measured in
[OVERNIGHT_BASELINE.md](OVERNIGHT_BASELINE.md).

The original model leaked future tokens during training and evaluation. Its
historical perplexity of about 6.3 is not a valid benchmark for the corrected
model. Legacy data and weights must be regenerated; the transferred version-2
datasets and completed version-2 Colab run already use the corrected workflow.
Legacy checkpoints are
rejected rather than silently interpreted as the new architecture.

For the configured RTX 4060 laptop environment and a one-command training launch,
see [LOCAL_TRAINING.md](LOCAL_TRAINING.md).
For a Colab T4 upload bundle and resumable Drive checkpoints, see
[COLAB_TRAINING.md](COLAB_TRAINING.md).
The fixed baseline architecture, completed Colab run, and evaluation protocol are
documented in [BASELINE.md](BASELINE.md).
For moving the project to another PC and continuing with Codex, see
[TRANSFER_README.md](TRANSFER_README.md) and [CODEX_HANDOFF.md](CODEX_HANDOFF.md).

## Training mathematics

### Reply-only loss and perplexity

Let $x_{i,1:T_i}$ be tokenized conversation $i$. Define $m_{i,t}=1$ for a
supervised reply token, including its terminal EOS, and $m_{i,t}=0$ for context,
padding or other unsupervised positions. With natural logarithms and at least
one supervised target, the token-normalized negative log-likelihood is

$$
\mathcal L_{\mathrm{SFT}}(\theta)
=-\frac{\sum_i\sum_{t=2}^{T_i}m_{i,t}\log p_\theta(x_{i,t}\mid x_{i,<t})}
{\sum_i\sum_{t=2}^{T_i}m_{i,t}},
\qquad \mathrm{PPL}=\exp(\mathcal L_{\mathrm{SFT}}).
$$

The hidden state at position $t-1$ predicts token $t$: shift exactly once.
Masked context tokens still condition the reply, but contribute no direct target
loss. The same mask convention is applied to the selected assistant spans in
whole-conversation SFT. Ignored labels use `-100`; see the
[PyTorch cross-entropy definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

If batch $b$ has mean loss $L_b$ and $N_b$ supervised tokens, aggregation is
$L=\sum_b N_bL_b/\sum_b N_b$, followed by $\mathrm{PPL}=e^L$.
This follows by adding token loss sums before dividing; averaging batch
perplexities generally gives a different answer. Our accumulation groups use
this token weighting too. Lower perplexity measures better prediction on the
scored tokens, not guaranteed role consistency or conversational quality;
different tokenizers do not provide directly comparable token perplexities.

Implementation: [scratch-model loss and evaluation](mini_deepseek.py),
[LoRA target masking](posttraining/lora_data.py) and
[token-weighted SFT](posttraining/lora_train.py).

### LoRA: adapting a frozen weight matrix

For a frozen matrix $W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$,
LoRA learns two smaller matrices:

$$
W=W_0+\Delta W,
\qquad \Delta W=\frac{\alpha}{r}BA,
\qquad A\in\mathbb R^{r\times d_{\mathrm{in}}},
\quad B\in\mathbb R^{d_{\mathrm{out}}\times r}.
$$

Here $r$ is the adapter rank and $\alpha/r$ scales its update;
$\operatorname{rank}(\Delta W)\le r$. Counting entries gives
$r(d_{\mathrm{in}}+d_{\mathrm{out}})$ trainable parameters instead of
$d_{\mathrm{in}}d_{\mathrm{out}}$. For a square matrix of width $d$, the ratio
is $2r/d$. This is a parameter-count ratio, not a total VRAM ratio: frozen base
weights, activations and runtime buffers still occupy memory.
[Hu et al., LoRA (2021), Section 4.1](https://arxiv.org/abs/2106.09685).

Our S1 configuration uses $r=16$, $\alpha=32$, and attention `q_proj`, `k_proj`,
`v_proj`, `o_proj` adapters. The equation describes the effective linear weight
at inference; S1 training additionally uses adapter dropout. See the
[S1 configuration](configs/ministral_s1_lora.json) and
[adapter training implementation](posttraining/lora_train.py).

### DPO: learning from preferred and rejected replies

Let $x$ be a prompt, $y^+$ the preferred reply, $y^-$ the rejected reply,
$\pi_\theta$ the trainable policy, and $\pi_{\mathrm{ref}}$ the frozen SFT reference.
The sequence log-probability is
$\ell_\theta(y\mid x)=\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t})$.
Our completion includes EOS and excludes prompt/padding loss; it is a sum,
not a length-normalized mean. Define the reference-relative preference margin:

$$
\Delta_\theta=
\big[\ell_\theta(y^+\mid x)-\ell_\theta(y^-\mid x)\big]
-\big[\ell_{\mathrm{ref}}(y^+\mid x)-\ell_{\mathrm{ref}}(y^-\mid x)\big].
$$

For $\beta>0$ and sigmoid $\sigma(z)=1/(1+e^{-z})$, standard DPO minimizes

$$
\mathcal L_{\mathrm{DPO}}(\theta)
=-\mathbb E_{(x,y^+,y^-)\sim\mathcal D}
\left[\log\sigma(\beta\Delta_\theta)\right].
$$

**Derivation sketch.** For a fixed prompt and reward $R$, the
KL-regularized objective
$\max_\pi\{\mathbb E_{y\sim\pi}[R(x,y)]-\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})\}$
has solution $\pi^*(y\mid x)=\pi_{\mathrm{ref}}(y\mid x)e^{R(x,y)/\beta}/Z(x)$,
assuming reference support and finite normalization. Rearranging gives
$R(x,y)=\beta\log[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]+\beta\log Z(x)$.
In the Bradley-Terry model, preference probability is
$\sigma(R(x,y^+)-R(x,y^-))$. The two $\log Z(x)$ terms cancel; parameterizing
the policy yields the loss above.
[Rafailov et al., DPO (2023), Section 4 and Appendix A](https://arxiv.org/abs/2305.18290).

Locally, each pair receives equal weight. When policy equals reference,
$\Delta_\theta=0$ and the loss is $\log 2$: our smoke check verifies this.
The [DPO core](posttraining/dpo_core.py) uses a frozen reference and disables
dropout. Our preference labels are assistant-authored/curated, not independent
human judgments. A lower preference loss does not establish better factual
replies; see the [matched bilingual comparison](BILINGUAL_CONTROL_RESULTS.md).

## Setup

Python 3.10 or newer, from this repository in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use `.venv/bin/python` instead. For NVIDIA GPU training, install a
CUDA-enabled PyTorch build. For this AMD PC, use the ROCm setup and launcher in
[AMD_TRAINING.md](AMD_TRAINING.md). `--device auto` selects
CUDA when available and otherwise CPU. Full training on CPU can be slow.

## Prepare a fresh dataset

```powershell
.\.venv\Scripts\python.exe prepare_data.py --output-dir data --context-length 768 --target-length 256
```

Downloads the Bluemoon roleplay dataset
(`rickRossie/bluemoon_roleplay_chat_data_300k_messages`) and GPT-2 tokenizer from
Hugging Face. Initial downloads need internet access. Threads are sorted by parsed
message timestamps, filtered to at least four nonempty messages, and split into
approximately **70% train / 10% validation / 20% test**, using seed 42. Entire
threads stay within one split. With small datasets, at least one thread is reserved
for each held-out split.

Each example contains three prior messages and the next reply. The script writes
raw and tokenized versions for all three splits under `data/`, for example
`data/bluemoon_train_ds` and `data/bluemoon_train_tok_ds`. Existing outputs are never
overwritten; select a new `--output-dir` when rebuilding.

Training and generation use the same formatting: messages separated by two
newlines, then an EOS boundary before the reply. Targets end with EOS. Truncation
keeps the most recent context. Padding is added only when batching; real EOS tokens
remain visible even though GPT-2 uses the same token ID for padding. Context and
target limits include their boundary/EOS tokens.

Each tokenized dataset includes a format version and tokenizer fingerprint.
Datasets generated by the old code must be rebuilt, not relabeled as version 2.

## Train a dense baseline

```powershell
.\.venv\Scripts\python.exe train.py --config configs/small.json --data-dir data/bluemoon_train_tok_ds --validation-dir data/bluemoon_validation_tok_ds --output-dir checkpoints/dense --epochs 5
```

`configs/small.json` uses four layers, 256 hidden dimensions, four attention heads,
and a 1,024-token context. The preparation limits above fit this model. Without
`--config`, the model uses 10 layers, 768 hidden dimensions, 12 heads, and a
3,072-token context. In either case, maximum context plus target lengths must fit
the configured model context length.

Training defaults are micro-batch 4, accumulation 8, learning rate 0.00015, and
1,000 warm-up updates. Warm-up is capped below the total number of updates for
short runs. Loss is normalized by the number of target tokens across each
accumulation group, including the final partial group. GPU BF16 is used when
supported natively on NVIDIA; older CUDA GPUs and ROCm auto mode use FP16 with
gradient scaling, and CPU uses float32. Use `--precision` to select explicitly. Only target-token NLL is reported as the
language-model training metric; the optional MoE penalty is kept separate.

The output folder contains:

- `ckpt_ep1.pt`, etc.: weights from each completed epoch.
- `best.pt`: weights with the lowest validation perplexity, when validation is enabled.
- `config.json` and `tokenizer/`: the exact model configuration and tokenizer.
- `metrics.json`: per-epoch training NLL and optional validation perplexity.
- `latest.pt` and `previous.pt`: complete resumable training state, saved every
  250 optimizer updates by default (`--save-every`).
- `progress.json`: saved epoch, data position, update count, and FP16 skip count.

Back up the entire checkpoint folder. Weight writes use a temporary file followed
by replacement. New runs require a fresh output directory. To continue, repeat the
original training command with `--resume checkpoints/dense/latest.pt`, keeping the
same output directory, data, and training settings. This restores optimizer,
scheduler, scaler, RNG, and data position. `--max-updates N` saves and stops at that
total update count; omit it to continue. Checkpoints, datasets, and the virtual environment are ignored
by Git; no full trained chatbot is bundled with the repository.

## Evaluate and generate

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint checkpoints/dense/best.pt --data-dir data/bluemoon_test_tok_ds
.\.venv\Scripts\python.exe chat.py --checkpoint checkpoints/dense/best.pt --prompt "Once upon a time, a cat found a mysterious key."
```

Use validation for model selection and reserve test data for the final evaluation.
Perplexity is calculated only over reply tokens, including EOS, with a single
next-token shift. Context and padding labels do not contribute.

`chat.py` prints only the generated reply. Pass repeated `--context "previous
message"` arguments in chronological order for conversation history, or
`--full-text` to include the supplied conversation in the output. Sampling options
include `--temperature` (0 for greedy decoding), `--top-k`, `--top-p`,
`--repetition-penalty`, and `--max-new-tokens`. Prompt plus output must fit the model
context length. Run any script with `--help` for its arguments.

Low held-out perplexity alone does not establish chat quality. Inspect generated
replies on held-out prompts as well. Training this small model from scratch on a
roleplay corpus does not give it the capabilities of a pretrained general chatbot.

## Optional MoE comparison and Markov baseline

`configs/small_moe.json` matches the small dense configuration but uses four
experts with top-2 routing in its last two layers. Train it in a different output
folder with the same data and compare validation metrics and generated samples.
Its balancing objective has gradients into the router and ignores padded tokens.
There is no duplicate MTP head or sequence-compressed attention in either model.

```powershell
.\.venv\Scripts\python.exe baseline.py --train-dir data/bluemoon_train_ds --test-dir data/bluemoon_test_ds --prompt "Once upon a time"
```

The Markov baseline uses raw text. Its word-level score is not directly comparable
to the transformer's subword-token perplexity.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The offline suite checks causal invariance and zero future-position gradients,
padding isolation, actual router gradients, target/EOS alignment, accumulation
weighting, checkpointing gradients, sampling, and version compatibility. Evaluation
is compared against independent prefix-only token scoring. An end-to-end test
covers preparation, training, saving, reloading, evaluation, and generation.

A tiny model also learns two synthetic prompt/reply pairs and generates the exact
replies, including learned termination. This is a learning/inference correctness
check, not a claim about general conversation quality. Resume tests compare
uninterrupted and continued CPU runs exactly; an FP16/scaler test runs when CUDA
is available. All 41 tests passed under ROCm. The modern experiment completed
four epochs with best validation perplexity 38.10; its fixed generation panel
revealed substantial coherence and relevance limitations. See
[GENERATION_EVALUATION.md](GENERATION_EVALUATION.md) for the measured results.

## Source layout

| File | Purpose |
| --- | --- |
| `mini_deepseek.py` | Corrected decoder, optional MoE, evaluation, generation |
| `training.py` | Mixed precision, target-weighted accumulation, resumable checkpoints |
| `conversation.py` | Shared formatting, token boundaries, and dataset metadata |
| `prepare_data.py` | Data preparation and thread-level splits |
| `train.py`, `evaluate.py`, `chat.py` | Command-line entry points |
| `evaluate_generation.py` | Fixed validation prompts, seeded replies, repetition metrics, human review CSV |
| `workflow.py` | Versioned model configuration and checkpoint loading |
| `baseline.py`, `markov_baseline.py` | Original Markov baseline |
| `configs/` | Small dense and MoE model configurations |
| `build_colab_bundle.py` | Portable Colab notebook and source/data ZIP builder |
| `tests/` | Offline regression and learning checks |
| `notebooks/chatbot_V1.ipynb` | Archived original notebook; execution outputs removed |

The original PDF and architecture diagram are retained as historical artifacts;
they do not describe the corrected architecture.

Original project: Zhengyi Chen, May 2025.
