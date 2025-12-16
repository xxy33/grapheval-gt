# GraphEval-GT

 **GraphEval-GT: Enhancing Viewpoint-Based Abstract Evaluation with Graph Transformer Mechanisms**

This repository contains the official implementation of **GraphEval-GT**, a framework for automated scientific idea evaluation. It constructs a **Structured Viewpoint Graph (SVG)** from abstracts and applies a global Structurally-Aware Graph Attention Network.

## 📂 Directory Structure

```text
.
├── main.py                       # Unified CLI entry point
├── config.yaml                   # (Expected) Global Hyperparameter configuration
├── src/                          # Core source code
│   ├── model_arch.py             # Definition of GraphEval-GT Network & Attention
│   ├── trainer.py                # Training loop implementation
│   ├── evaluator.py              # Validation and testing logic
│   ├── data_preprocessor.py      # Graph construction from extracted viewpoints
│   ├── azure_llm_api.py          # LLM API wrapper (Azure OpenAI) for viewpoint extraction
│   └── config_loader.py          # Configuration parser
├── script/                       # Data processing pipeline scripts
│   ├── 01_extract_all_viewpoints.py  # Step 1: Query LLM to extract raw viewpoints
│   ├── 02_fix_unicode_escapes.py     # Step 2: Clean text formatting/encoding
│   └── 03_jsonl_concat.py            # Step 3: Merge processed data chunks
└── requirements.txt              # Python dependencies
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/grapheval-gt.git
   cd grapheval-gt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

We provide a unified entry point `main.py` for training and evaluation, while data preparation is handled via scripts.

### 1. Data Pipeline (Preprocessing)

Before training, raw text needs to be processed into viewpoint graphs. Follow the numbered scripts in `script/`:

**Step 1: Extract Viewpoints**
Queries the LLM to decompose abstracts into structured Viewpoints.
```bash
python script/01_extract_all_viewpoints.py
```

**Step 2: Data Cleaning & Merging**
Fix encoding issues and concatenate distributed files into a single dataset.
```bash
python script/02_fix_unicode_escapes.py
python script/03_jsonl_concat.py
```

**Step 3: Graph Construction**
Convert the cleaned JSONL data into PyG/DGL graph objects (Saved to paths defined in `config`):
```bash
python -m src.data_preprocessor
```

### 2. Training

To train the GraphEval-GT model using the processed graphs:

```bash
# Train on ICLR Papers dataset (Default)
python main.py train --dataset iclr_papers

# Train on AI Researcher dataset
python main.py train --dataset ai_researcher
```

### 3. Evaluation

To evaluate a trained model on the test set:

```bash
python main.py evaluate --dataset iclr_papers
```

## ⚙️ Configuration

Hyperparameters, file paths, and API settings are managed via `src/config_loader.py`. Please ensure you have set up your data paths correctly.

**LLM Setup:**
Since `src/azure_llm_api.py` is used, ensure your Azure OpenAI or OpenAI API keys are set in your environment variables or config file before running Step 1.

## 🔗 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{chen2025graphevalgt,
  title={GraphEval-GT: Enhancing Viewpoint-Based Abstract Evaluation with Graph Transformer Mechanisms},
  author={Chen, Xiaoyu and Wu, Fengge and Zhao, Junsuo},
  booktitle={ICASSP 2025},
  year={2025}
}
```
