# mini DeepSeek: Compact Transformer with MLA & MoE

---

## Overview

mini DeepSeek is a pedagogical PyTorch implementation of a compact Transformer model designed for efficient language modeling. It integrates two core innovations from DeepSeek-V3:

- **Multi-Head Latent Attention (MLA):** A Linformer-inspired low-rank projection that reduces attention complexity from \(O(T^2d)\) to \(O(TLd)\), cutting FLOPs by ~75% at sequence length 1024.
- **Sparse Mixture-of-Experts (MoE):** A top-2 gating mechanism over four experts, doubling model capacity for only 2× compute, with a load-balancing auxiliary loss.

This repository provides the full training pipeline, from data preprocessing to model evaluation, along with scripts for experimentation.

---

## Features

- **Model Architecture**
  - 12-layer, pre-norm Transformer.
  - Early layers: MLA + Feed-Forward.
  - Later layers: MLA + MoE.
- **Tokenizer**
  - Byte-level BPE using Hugging Face GPT-2 tokenizer.
  - Full UTF-8 support; preserves whitespace.
- **Scalable Training**
  - Mixed-precision (BF16) on NVIDIA A100 GPUs.
  - Micro-batching with gradient accumulation.
  - AdamW optimizer with cosine-decay learning rate and warm-up.
- **Eval & Metrics**
  - Validation perplexity ~6.3 on the Bluemoon Roleplay Chat test split.
  - Memory footprint ~18 GB peak; codebase under 1 000 lines.

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (1.12+)
- Transformers & Datasets (Hugging Face)
- NVIDIA CUDA toolkit

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/mini-deepseek.git
cd mini-deepseek

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. Download or prepare the Bluemoon Roleplay Chat corpus.
2. Place the cleaned threads (≥4 messages each) in `data/roleplay/`.
3. Run preprocessing:

```bash
python scripts/preprocess.py   --input_dir data/roleplay/   --output_dir data/processed/   --window_size 4
```

---

## Training

```bash
python train.py   --train_data data/processed/train.jsonl   --val_data data/processed/val.jsonl   --model_config configs/mlamoe_config.yaml   --batch_size 8   --accumulate_steps 4   --max_tokens 4096   --fp16   --lr 1e-4   --warmup_steps 1000   --output_dir checkpoints/
```

- **Checkpointing:** Saved every epoch in `checkpoints/`.
- **Logging:** TensorBoard logs in `runs/`.

---

## Evaluation

```bash
python evaluate.py   --model_path checkpoints/epoch_last   --test_data data/processed/test.jsonl   --output metrics.json
```

- Reports validation perplexity and inference throughput.

---

## Project Structure

```
mini-deepseek/
├── configs/            # Model & training configurations
├── data/               # Raw and processed datasets
├── scripts/            # Preprocessing & utility scripts
├── src/                # Model implementation
├── train.py            # Training entrypoint
├── evaluate.py         # Evaluation entrypoint
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and instructions
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions or contributions, please open an issue or reach out to Zhengyi Chen at <zhengyi.chen@example.com>.
