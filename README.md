# Agentic AI: Next-Gen 5G Networking with Privacy-Aware Intrusion Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-IEEE%20DataPort-red.svg)](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://github.com/slundberg/shap)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-blue.svg)](https://www.langchain.com/)

A comprehensive implementation of **Agentic AI for next-generation 5G networking**, featuring Vertical Federated Learning (VFL) for privacy-aware network intrusion detection, SHAP-based explainability, and intelligent agentic reasoning for automated mitigation recommendations.

## 📋 Overview

This project implements a **next-generation 5G networking security system** powered by **Agentic AI** and Vertical Federated Learning (VFL). The system enables privacy-preserving collaborative learning across distributed network tiers (RAN, Edge, Core) while maintaining data locality. By partitioning network flow features across three logical agents representing different network domains, the system simulates realistic 5G network observability constraints while enabling intelligent, explainable intrusion detection and automated response orchestration.

| Theme | What it delivers |
|-------|------------------|
| **Next-gen 5G networking** | Agentic AI aligned with distributed 5G architectures (RAN, Edge, Core). |
| **Vertical feature partitioning** | Flow features split across three agents by realistic observability boundaries. |
| **Privacy-preserving VFL** | Joint training without exchanging raw data between tiers or parties. |
| **SHAP explainability** | Per-agent attributions to support mitigation and audit reasoning. |
| **Agentic reasoning + RAG** | LLM action plans grounded in retrieval over policies and knowledge documents. |
| **Performance** | >96% test accuracy on standardized intrusion-detection benchmarks. |

## 📁 Project Structure

```
AgenticAI/
├── datasets/              # Dataset CSV files
├── model/                 # Trained models (created after training)
│   ├── vfl_model_best.pth
│   ├── meta_model_best.pth
│   ├── model_metadata.json
│   └── scaler*.pkl
├── utils/                 # Core modules
│   ├── __init__.py
│   ├── model_utils.py    # VFL / meta-model PyTorch modules
│   ├── vfl_utils.py      # Utility functions
│   └── rag_utils.py      # RAG: predictions/config, FAISS save/load, action-plan JSON
├── RAG_docs/             # RAG knowledge base
│   ├── agentic_features.json
│   ├── attack_options.json
│   ├── knowledge/        # PDF knowledge base
│   └── vector_store/     # FAISS vector store
├── outputs/              # Training outputs
├── inputs/               # Input data for prediction
├── Documents/            # Project documentation and guides
├── Train.ipynb
├── Predict.ipynb
├── Index.ipynb
├── Plan.ipynb
├── Score.ipynb
└── requirements.txt
```

## 🚀 Quick Start

### System Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS 10.14+
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: ~2GB free space
- **GPU**: Optional (NVIDIA GPU with CUDA 11.8+ for acceleration)

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd AgenticAI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install PyTorch with GPU support (optional)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 6. Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Dataset Setup

**Quick Setup**:
```bash
# Create datasets directory
mkdir datasets

# Download from IEEE DataPort (requires subscription/login)
# After downloading, extract and locate: undersampled_CIC2017_dataset.csv
# Copy to datasets/ directory:
# Windows:
copy "path\to\undersampled_CIC2017_dataset.csv" datasets\
# Linux/macOS:
cp path/to/undersampled_CIC2017_dataset.csv datasets/

# Verify
dir datasets\*.csv  # Windows
ls datasets/*.csv   # Linux/macOS
```

**Note**: IEEE DataPort subscription required. See [Dataset section](#-dataset) below for detailed information and download instructions.

### Configuration

Create `.env` file for OpenAI API (required for action plan generation):
```env
OPENAI_API_KEY=sk-your-api-key-here
```

**Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)

### Verification

Test your setup:
```python
# Test imports
python -c "from utils.model_utils import VFLModel; from utils.vfl_utils import load_agent_definitions; print('✓ Setup OK')"

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📓 Notebooks Workflow

Execute notebooks in this order:

```
Train.ipynb  →  Predict.ipynb  →  Index.ipynb  →  Plan.ipynb  →  Score.ipynb
```

### 1. Train.ipynb (Training)

**Purpose**: Train VFL and standard neural network baseline models

**Why**: Builds the core intrusion detection models with privacy-preserving federated learning

**What it does**:
- Loads `undersampled_CIC2017_dataset.csv` from `datasets/` directory
- Partitions features into RAN (23), Edge (32), Core (42) agents
- Trains VFL model with vertical federated learning
- Trains standard neural network baseline for comparison
- Trains meta-model for efficient SHAP computation
- Computes SHAP values for explainability
- Generates agent-level attributions and mitigation recommendations

**Inputs**:
- `undersampled_CIC2017_dataset.csv` in `datasets/` directory
- `RAG_docs/agentic_features.json` (agent definitions)

**Outputs**: 
- `model/vfl_model_best.pth` - Trained VFL model
- `model/meta_model_best.pth` - Meta-model for SHAP
- `model/model_metadata.json` - Model configuration
- `model/scaler*.pkl` - Feature scalers
- `model/shap_background.npy` - SHAP background data
- `outputs/vfl_shap_performance_*.csv` - Performance metrics
- `outputs/model_comparison_*.json` - Model comparison results

**Execution Time**: 30-60 minutes (depending on hardware)

**Configuration**:
```python
embed_dim = 64
hidden_dim = 128
num_classes = 9
epochs = 100
learning_rate = 0.001
```

**Run**:
```bash
jupyter notebook Train.ipynb
```

---

### 2. Predict.ipynb (Inference)

**Purpose**: Generate predictions for new network flows

**Why**: Apply trained models to new data and get explainable predictions

**What it does**:
- Loads trained models from `model/` directory
- Processes new network flow data
- Generates predictions with confidence scores
- Computes SHAP values for each prediction
- Identifies dominant agent per prediction

**Prerequisites**: Must run `Train.ipynb` first

**Inputs**:
- Trained models in `model/` directory
- Input CSV file with network flow features (place in `inputs/` directory)
- Must have 88 features matching training data

**Outputs**: 
- `RAG_docs/predictions/predictions_*.json` - Predictions with SHAP values
- Console output with prediction details

**Execution Time**: 1-5 minutes (depending on sample size)

**Run**:
```bash
jupyter notebook Predict.ipynb
```

---

### 3a. Index.ipynb (RAG index)

**Purpose**: Ingest the knowledge base and persist a FAISS vector store.

**What it does**:
- Loads JSON/PDF documents from `RAG_docs/knowledge/` (load helpers live in this notebook)
- **PDFs:** Per-page PDF rows are merged into **sectioned documents** (heading heuristics), then each section is split into **semantic parents** (paragraph-embedding similarity), then into **child** chunks — see `utils/rag_index_build.py`
- **Retrieval titles:** **One search-oriented title per semantic parent** (default: local HF summarizer; optional extractive keyword line or LLM via `TITLE_MODE`). Child vectors embed **`retrieval_title + child text`** for better query overlap
- Embeds with SentenceTransformers (`all-MiniLM-L6-v2`), builds FAISS via batched helpers in `utils/rag_utils.py` / `utils/rag_index_build.py`
- Persists via `utils/rag_utils.save_vector_store` to `RAG_docs/vector_store/` and **`utils/rag_index_build.save_parent_store`** → **`rag_parents.json`** (full parent text + titles for Part 2)

**Prerequisites**: Knowledge files in `RAG_docs/knowledge/`

**Run** (whenever knowledge files change):
```bash
jupyter notebook Index.ipynb
```

---

### 3b. Plan.ipynb (Action planning)

**Purpose**: Generate context-aware mitigation action plans from SHAP predictions.

**Why**: Converts SHAP attributions into actionable security recommendations using RAG + OpenAI, and runs a **six-mode retrieval / reranking ablation** so `Score.ipynb` can compare Stage-1 retrievers and Stage-2 rerankers head-to-head.

**What it does**:
- **Loads** the saved FAISS index and parent store from Part 1 only (does not re-read `RAG_docs/knowledge/`)
- Generates retrieval queries per sample (deterministic template, deterministic rephrase, optional LLM variants; configurable)
- For every sample, assembles context under **six retrieval / reranking modes** — all two-stage modes use a **100-chunk Stage-1 shortlist** and select the **top 10** for the LLM:

  | Mode | Stage 1 (100) | Stage 2 (10) |
  |---|---|---|
  | `none` (Dense MMR) | Dense FAISS multi-query | MMR |
  | `bm25` | BM25 over full child corpus | truncate |
  | `doct5query` | BM25 over doc2query-expanded corpus (`castorini/doc2query-t5-base-msmarco`, 20 synthetic queries / chunk) | truncate |
  | `crossencoder` (Dense → CE) | Raw dense FAISS top-100 (no MMR) | CrossEncoder |
  | `bm25_crossencoder` (BM25 → CE) | BM25 top-100 | CrossEncoder |
  | `colbert` (Dense → ColBERT) | Raw dense FAISS top-100 (no MMR) | ColBERT v2 late-interaction |

- Expands the top-10 child chunks to full parent sections via `rag_parents.json` and calls OpenAI once per mode per sample
- Persists each sample's full comparison (top-10 + 100-chunk candidate pool + LLM output for every mode) via `utils.rag_utils.save_rerank_comparison_report`
- Caches the doc2query-expanded corpus to `RAG_docs/rag_parents_doc2query.json` so subsequent runs skip the T5 generation step

**Shared code**: `utils/rag_utils.py` (predictions/config loaders, FAISS save/load + manifest, party/tier helpers, rerank-comparison report writer). KB file loading and index build are in Part 1; Part 2 only loads the vector store.

**Prerequisites**:
- Part 1 completed (`RAG_docs/vector_store/` exists)
- Predictions from `Predict.ipynb`
- OpenAI API key in `.env`
- `RAG_docs/attack_options.json`, `RAG_docs/agentic_features.json`

**Inputs**:
- `RAG_docs/predictions/*.json` - Predictions with SHAP
- `RAG_docs/vector_store/` - FAISS index + `rag_parents.json` from Part 1

**Outputs**:
- `RAG_docs/action_plans/rerank_comparison_sample_<id>_<ts>.json` — one per sample, containing all six modes' top-10, 100-chunk candidate pool, and LLM action plan
- `RAG_docs/rag_parents_doc2query.json` — cached doc2query-expanded corpus

**Run**:
```bash
jupyter notebook Plan.ipynb
```

---

### 4. Score.ipynb (Evaluation)

**Purpose**: Evaluate the six retrieval / reranking modes from `Plan.ipynb` using BEIR / CRAG-aligned metrics and produce per-attack-class tables + grouped bar charts.

**Why**: Validates which Stage-1 retriever × Stage-2 reranker combination delivers the best retrieval quality, context faithfulness, and downstream LLM reasoning.

**What it does**:
- Loads every `rerank_comparison_sample_*.json` produced by `Plan.ipynb` (each file contains all six modes' top-10 + 100-chunk candidate pool + LLM output for one sample)
- Builds a deterministic **label-derived reference string** per sample from `attack_options.json` (actions per attack label) and `agentic_features.json` (which agents execute which actions)
- Derives **adaptive thresholds** per batch (median / 25th-percentile of the observed score distribution, clamped to relaxed floors) for Recall@k relevance and the CRAG proxy
- Computes seven metrics per sample per mode and aggregates them by rerank mode and by attack class
- Exports a scoring summary (`RAG_docs/scoring/scoring_results.json`) and grouped bar charts (six bars per metric group)

**Prerequisites**: Rerank comparison reports from `Plan.ipynb`

**Inputs**:
- `RAG_docs/action_plans/rerank_comparison_sample_*.json` - Per-sample comparison reports (all six modes)
- `RAG_docs/attack_options.json` - Reference actions
- `RAG_docs/agentic_features.json` - Agent definitions

**Outputs**:
- `RAG_docs/scoring/scoring_results.json` - Evaluation results with metric notes
- Per-rerank-mode and per-attack-class tables (one six-row table per attack class)
- Grouped bar charts (six bars per metric group, one group per metric)

**Execution Time**: 2-10 minutes (depending on number of samples)

**Metrics** (seven per sample × six modes):
1. **Recall@10** — graded-relevance recall of the top-10 vs. the 100-chunk candidate pool, using `0.5·ROUGE-1 + 0.5·SBERT` with an adaptive relevance threshold.
2. **nDCG@10** — standard nDCG using the same graded relevances over the top-10.
3. **RI_SBert** — SBERT cosine between retrieved top-10 context and the reference text (context grounding).
4. **CRAG** — 3-valued CRAG-style truthfulness proxy rescaled to `[0, 1]` (`1` accurate / `0.5` missing or partial / `0` incorrect) between retrieved context and reference text.
5. **Util_SBert / Util_CRAG** — same pair of metrics, but between retrieved context and the LLM's `overall_reasoning` (how much of the retrieved evidence the LLM actually grounded its answer in).
6. **LLM_SBert / LLM_CRAG** — same pair, but between the LLM's `overall_reasoning` and the reference text.
7. **BertscoreF1** — BERTScore F1 between the ground-truth reference statement and the LLM's `overall_reasoning`.

**Run**:
```bash
jupyter notebook Score.ipynb
```

## 🛠️ Technologies & Tools

### Core ML/DL Framework
- **PyTorch** (≥2.0.0) - Deep learning framework
- **NumPy** (≥1.23.0) - Numerical computing
- **Pandas** (≥1.5.0) - Data manipulation
- **scikit-learn** (≥1.2.0) - Machine learning utilities

### Explainability
- **SHAP** (≥0.42.0) - SHapley Additive exPlanations for model interpretability

### LLM & RAG
- **OpenAI API** - GPT-4 / GPT-3.5 for action plan generation
- **LangChain** (≥0.3.0) - LLM application framework
- **FAISS** - Dense vector similarity search (Stage-1 retrieval for dense modes)
- **Sentence Transformers** (≥2.2.0) - Bi-encoder embeddings + CrossEncoder reranker
- **rank-bm25** - Lexical Stage-1 retriever (`bm25` and `bm25_crossencoder` modes)
- **Hugging Face Transformers** + **SentencePiece** - T5 (`castorini/doc2query-t5-base-msmarco`) for `doct5query` corpus expansion
- **RAGatouille** - ColBERT v2 late-interaction reranker (with a patched CPU MaxSim kernel for Windows)

### Evaluation
- **BERTScore** (≥0.3.13) - Contextual similarity between LLM reasoning and reference text
- **ROUGE** (≥1.0.1) - Lexical overlap used inside the graded-relevance score
- **NLTK** (≥3.8.0) - Natural language processing
- **CRAG-style proxy** - 3-valued truthfulness rescaled to `[0, 1]` (1 accurate / 0.5 missing or partial / 0 incorrect), with adaptive per-batch thresholds

### Data Processing
- **PyPDF2** / **pypdf** - PDF document processing for knowledge base

### Development
- **Jupyter Notebook** - Interactive development environment
- **Python 3.8+** - Programming language

## 📊 Dataset

### UM-NIDS Dataset (Unified Multimodal Network Intrusion Detection Systems Dataset)

**Source**: [IEEE DataPort - UM-NIDS Dataset](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset)

**Description**: 
The Unified Multimodal Network Intrusion Detection System (UM-NIDS) dataset is a comprehensive, standardized dataset that integrates network flow data from four well-established datasets: CIC-IDS 2017, CIC-IoT 2023, UNSW-NB15, and CIC-DDoS 2019. This project uses the **undersampled version of CIC-IDS 2017** component from the UM-NIDS dataset.

**Dataset File Used**: `undersampled_CIC2017_dataset.csv`

**Key Statistics**:
- **Original Features**: 120 features (from CIC-IDS 2017)
- **Processed Features**: 88 features (32 removed for 5G observability constraints)
- **Total Records**: ~390,000 labeled flows
- **Data Split**: 64% train, 16% validation, 20% test
- **Attack Classes**: 9 classes
  - BENIGN (normal traffic)
  - BOT, DDOS, DOS
  - FTPPATATOR, SSHPATATOR
  - PORTSCAN, WEBATTACK
  - OTHERS (combined rare attacks)

**Download**: 
1. Visit [IEEE DataPort UM-NIDS Dataset](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset)
2. Login/Subscribe to IEEE DataPort (IEEE members have free access)
3. Download "Undersampled version of all four datasets with same feature space" (394.19 MB)
4. Extract and locate `undersampled_CIC2017_dataset.csv`
5. Place in `datasets/` directory

**Citation**:
```
Irfan Khan, Nathaniel Bastian, Syed Wali, Yasir Ali Farrukh, 
"Unified Multimodal Network Intrusion Detection Systems Dataset", 
IEEE Dataport, October 2, 2024, doi:10.21227/d8at-gb29
```

**Original CIC-IDS 2017 Citation** (for reference):
```
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, 
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", 
4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, 2018
```

## 📈 Results

### Model Performance

| Model | Accuracy | Macro Recall | Macro F1 |
|-------|----------|--------------|----------|
| **VFL** | 0.9617 | 0.9864 | 0.8594 |
| **Standard NN** | 0.9643 | 0.9863 | 0.8734 |

**Key Finding**: VFL achieves comparable performance (<0.3% difference) while maintaining data privacy.

### Agent Contributions (SHAP Analysis)

- **RAN**: 32.14% mean contribution
- **Edge**: 39.25% mean contribution  
- **Core**: 28.61% mean contribution

## 🔧 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure you're in project root
cd /path/to/AgenticAI

# Verify utils directory exists
# Reinstall if needed
pip install -r requirements.txt
```

**CUDA/GPU Issues**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
# If False, reinstall PyTorch with CUDA support (see Installation section)
```

**Dataset Not Found**:
- Verify CSV files are in `datasets/` directory
- Check file names (should be `.csv` extension)
- Ensure running notebook from project root

**OpenAI API Errors**:
- Verify `.env` file exists in project root
- Check API key format: `OPENAI_API_KEY=sk-...`
- Ensure no extra spaces or quotes
- Test: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OK' if os.getenv('OPENAI_API_KEY') else 'Missing')"`

**Memory Issues**:
- Reduce batch size in training notebook
- Use smaller dataset subset for testing
- Close other applications
- Consider using CPU if GPU memory is limited

**NLTK Data Missing**:
```python
import nltk
nltk.download('punkt')
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/AgenticAI.git
   cd AgenticAI
   git remote add upstream https://github.com/original-owner/AgenticAI.git
   ```

2. **Create Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**:
   - Follow PEP 8 style guidelines
   - Add type hints to functions
   - Include docstrings (Google-style)
   - Update documentation as needed

4. **Test Changes**:
   ```bash
   python -c "from utils.model_utils import VFLModel; print('✓ Imports OK')"
   ```

5. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Add: brief description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request** on GitHub

### Coding Standards

- **Style**: PEP 8 (4 spaces, max 100 chars per line)
- **Naming**: `snake_case` for functions, `PascalCase` for classes
- **Type Hints**: Include for all function signatures
- **Docstrings**: Google-style docstrings
- **Formatting**: Use `black` for code formatting

### Reporting Bugs

Include:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, package versions)
- Error messages and stack traces

### Suggesting Enhancements

Include:
- Clear description
- Use case and rationale
- Proposed solution (if available)
- Alternatives considered

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **UM-NIDS Dataset**: [IEEE DataPort - Unified Multimodal Network Intrusion Detection Systems Dataset](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset) (doi:10.21227/d8at-gb29)
- **CICIDS-2017 Dataset**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) (original source)
- **SHAP**: [SHapley Additive exPlanations](https://github.com/slundberg/shap)
- **LangChain**: [LangChain Framework](https://www.langchain.com/)
- **PyTorch**: [PyTorch Deep Learning Framework](https://pytorch.org/)

---

**Note**: This project is for research and educational purposes. Ensure proper data handling and compliance with privacy regulations when deploying in production.
