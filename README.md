# CyberRerankBench

**Evaluating rerankers for cybersecurity RAG in an agentic VFL-NIDS pipeline**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-CICIDS--2017-red.svg)](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple.svg)](https://github.com/slundberg/shap)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-blue.svg)](https://www.langchain.com/)

Retrieval-Augmented Generation (RAG) improves Large Language Model (LLM) reliability in cybersecurity by grounding responses in external knowledge. Effectiveness depends on retrieved context quality, where **reranking models** play a key role. Rerankers have not been systematically evaluated in cybersecurity-specific settings, which involve noisy logs, semantically overlapping attack patterns, and policy-driven reasoning.

**CyberRerankBench** is a benchmark for evaluating rerankers in cybersecurity RAG systems. It is implemented in an agentic Network Intrusion Detection System (NIDS) pipeline that integrates:

- **Vertical Federated Learning (VFL)** for data-local detection across distributed enterprise domains
- **SHAP-based attribution** for explainability
- **Constrained LLM reasoning** grounded in a multi-framework cybersecurity knowledge base (MITRE ATT&CK, NIST SP 800-53, CIS Controls v8, ISO/IEC 27001)

Retrieval queries are built dynamically from detection outputs and attribution signals, so ranking strategies are scored under realistic, domain-aware conditions.

**Keywords:** information retrieval, Retrieval-Augmented Generation, reranking, benchmark evaluation, cybersecurity, intrusion detection

## Overview

| Theme | What it delivers |
|-------|------------------|
| **CyberRerankBench** | Head-to-head evaluation of retrieval and reranking under cybersecurity RAG constraints. |
| **Agentic NIDS pipeline** | Detection → attribution → retrieval query → ranked context → constrained mitigation plan. |
| **Vertical feature partitioning** | Flow features stay local to each domain party; only embeddings are shared. |
| **Privacy-preserving VFL** | Joint classification without exchanging raw telemetry. |
| **SHAP explainability** | Per-party attributions that both explain detections and drive RAG queries. |
| **Policy-grounded RAG** | LLM action plans constrained by retrieved controls from ATT&CK, NIST, CIS, and ISO/IEC 27001. |

<p align="center">
  <img src="CyberRank.png" alt="VFL cybersecurity network topology: serial attack path across four observation points with secure VFL, SHAP, and RAG mitigation. Raw data never leaves each participant." width="900"/>
</p>

<p align="center"><em>Conceptual topology: serial attack path across observation points; embeddings only — raw data never leaves a participant. Detection, SHAP attribution, and RAG mitigation run in the coordination layer over a multi-framework knowledge base.</em></p>

```mermaid
flowchart LR
  subgraph L1[Multi-hop observations]
    ISP[ISP / upstream]
    RS[Router / switch]
    FW[Firewall / IDS]
    EDR[EDR platform]
  end

  VFL[Secure VFL + SHAP<br/>classify · explain signals]

  subgraph RAG[RAG retrieval layer]
    direction LR
    IN[Ingest]
    EM[Embed / index]
    SE[Search N ≫ K]
    RK[Rerank → Top-K]
    IN --> EM --> SE --> RK
  end

  LLM[Constrained LLM<br/>policy + context → mitigation]

  ISP --> VFL
  RS --> VFL
  FW --> VFL
  EDR --> VFL
  VFL --> RAG --> LLM
```

## Benchmark design

CyberRerankBench evaluates **ten retrieval configurations** across **three tracks**:

| Track | Configurations | What it isolates |
|-------|----------------|------------------|
| **Policy corpus** | 6 modes | Stage-1 retriever × Stage-2 reranker on the multi-framework policy KB. |
| **Domain-index ablation** | 2 modes | Effect of domain-aware indexing on ranking quality. |
| **Response playbook** | 2 modes | Ranking over operational playbook-style corpus vs. policy text. |

On **CICIDS-2017** the protocol reports **250 stratified inference scenarios per track** (**2,500 mode–scenario rows** total). Queries are constructed from VFL labels, confidence, and SHAP attributions (party contributions and top indicators per domain).

### Scoring protocol

Scoring uses a **three-layer** protocol:

1. **Headline MiniLM metrics** — retrieval quality, context faithfulness, utilization, and LLM-to-reference alignment (`Score.ipynb`)
2. **Parallel cyber-encoder grades** — CySecBERT, CTI-BERT, and SecureBERT
3. **Class- and encoder-conditioned analysis** — whether mode rankings hold across attack classes and grader encoders

Headline metrics (per sample × mode):

| Metric | Role |
|--------|------|
| **Recall@10** | Graded-relevance recall of the top-10 vs. the 100-chunk Stage-1 pool (`0.5·ROUGE-1 + 0.5·SBERT`) |
| **nDCG@10** | Ranking quality over the same graded relevances |
| **RI_SBert / CRAG** | Context grounding vs. a label-derived reference (actions grouped by capable party) |
| **Util_SBert / Util_CRAG** | How much retrieved evidence the LLM actually used |
| **LLM_SBert / LLM_CRAG / BERTScore-F1** | Reasoning quality vs. the same reference |

### Policy-corpus modes (this repository)

All two-stage modes use a **100-chunk Stage-1 shortlist** and keep the **top 10** for the LLM:

| Mode | Stage 1 (100) | Stage 2 (10) |
|------|----------------|--------------|
| `none` (Dense MMR) | Dense FAISS multi-query | MMR |
| `bm25` | BM25 over full child corpus | truncate |
| `doct5query` | BM25 over doc2query-expanded corpus (`castorini/doc2query-t5-base-msmarco`, 20 synthetic queries / chunk) | truncate |
| `crossencoder` (Dense → CE) | Raw dense FAISS top-100 (no MMR) | CrossEncoder |
| `bm25_crossencoder` (BM25 → CE) | BM25 top-100 | CrossEncoder |
| `colbert` (Dense → ColBERT) | Raw dense FAISS top-100 (no MMR) | ColBERT v2 late-interaction |

`Plan.ipynb` and `Score.ipynb` implement this six-mode policy-corpus ablation end to end.

## Findings

On CICIDS-2017, under the three-track protocol:

- On the **policy corpus** track, **dense MiniLM retrieval still leads on faithfulness**.
- The **response playbook** track **raises pool recall** but **weakens utilization** proxies (retrieved text is broader than what the LLM actually uses).
- **Cyber-encoder grades reorder modes** relative to MiniLM and expose **encoder–corpus mismatch**.
- **No single reranker dominates** across attack classes or encoders.
- **Class- and encoder-conditioned routing** is treated as a hypothesis for future work, not a settled conclusion.

## Project structure

```
VFL-RAG/
├── datasets/              # CICIDS-2017 CSV (not committed)
├── model/                 # Trained models (created after training)
│   ├── vfl_model_best.pth
│   ├── meta_model_best.pth
│   ├── model_metadata.json
│   └── scaler*.pkl
├── utils/                 # Core modules
│   ├── model_utils.py     # VFL / meta-model PyTorch modules
│   ├── vfl_utils.py       # Feature partition and VFL helpers
│   ├── rag_utils.py       # Predictions/config, FAISS, rerank reports
│   └── rag_index_build.py # Sectioned PDF ingest, parent/child index
├── RAG_docs/              # Knowledge base and RAG artifacts
│   ├── agentic_features.json
│   ├── attack_options.json
│   ├── knowledge/         # Policy PDFs (ATT&CK, NIST, CIS, ISO, …)
│   ├── vector_store/      # FAISS index + rag_parents.json
│   ├── predictions/       # Inference + SHAP JSON
│   ├── action_plans/      # Per-sample six-mode rerank reports
│   └── scoring/           # Aggregated metrics
├── outputs/               # Training comparison JSON
├── inputs/                # Input CSVs for prediction
├── Documents/             # Pipeline and RAG guides
├── docs/                  # Architecture diagrams
├── CyberRank.png          # Topology figure
├── Train.ipynb
├── Predict.ipynb
├── Index.ipynb
├── Plan.ipynb
├── Score.ipynb
└── requirements.txt
```

## Quick Start

### System Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS 10.14+
- **Python**: 3.8 or higher (3.12 recommended; ColBERT/`voyager` wheels are most reliable on 3.12)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: ~2GB free space
- **GPU**: Optional (NVIDIA GPU with CUDA 11.8+ for acceleration)

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd VFL-RAG

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

IEEE DataPort subscription required. See [Dataset](#dataset) below.

### Configuration

Create a `.env` file for OpenAI (required for action-plan generation):

```env
OPENAI_API_KEY=sk-your-api-key-here
```

**Get API Key**: [OpenAI Platform](https://platform.openai.com/api-keys)

### Verification

```python
# Test imports
python -c "from utils.model_utils import VFLModel; from utils.vfl_utils import load_agent_definitions; print('Setup OK')"

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Notebooks workflow

Execute notebooks in this order:

```
Train.ipynb  →  Predict.ipynb  →  Index.ipynb  →  Plan.ipynb  →  Score.ipynb
```

Detection and attribution produce the queries; indexing builds the policy corpus; planning runs the six-mode ablation; scoring computes headline MiniLM metrics.

### 1. Train.ipynb (Training)

**Purpose**: Train VFL and standard neural network baseline models.

**What it does**:
- Loads `undersampled_CIC2017_dataset.csv` from `datasets/`
- Partitions features into domain parties (RAN 23, Edge 32, Core 42 in the current agent JSON)
- Trains the VFL classifier with vertical federated learning
- Trains a standard neural network baseline
- Trains a meta-model for efficient SHAP computation
- Computes SHAP values and party-level attributions used later as retrieval signals

**Inputs**:
- `undersampled_CIC2017_dataset.csv` in `datasets/`
- `RAG_docs/agentic_features.json` (party definitions)

**Outputs**:
- `model/vfl_model_best.pth` — trained VFL model
- `model/meta_model_best.pth` — meta-model for SHAP
- `model/model_metadata.json` — model configuration
- `model/scaler*.pkl` — feature scalers
- `model/shap_background.npy` — SHAP background data
- `outputs/vfl_shap_performance_*.csv` — performance metrics
- `outputs/model_comparison_*.json` — model comparison results

**Execution Time**: 30–60 minutes (depending on hardware)

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

**Purpose**: Generate explainable predictions for new network flows.

**What it does**:
- Loads trained models from `model/`
- Processes new network flow data
- Generates predictions with confidence scores
- Computes SHAP values per prediction
- Identifies the dominant party per prediction (used in RAG query construction)

**Prerequisites**: Must run `Train.ipynb` first

**Inputs**:
- Trained models in `model/`
- Input CSV with network flow features in `inputs/` (same feature space as training)

**Outputs**:
- `RAG_docs/predictions/predictions_*.json` — predictions with SHAP values

**Execution Time**: 1–5 minutes (depending on sample size)

**Run**:
```bash
jupyter notebook Predict.ipynb
```

---

### 3a. Index.ipynb (RAG index)

**Purpose**: Ingest the multi-framework knowledge base and persist a FAISS vector store.

**What it does**:
- Loads JSON/PDF documents from `RAG_docs/knowledge/`
- **PDFs:** Per-page rows are merged into **sectioned documents** (heading heuristics), then each section is split into **semantic parents** (paragraph-embedding similarity), then into **child** chunks — see `utils/rag_index_build.py`
- **Retrieval titles:** One search-oriented title per semantic parent (default: local HF summarizer; optional extractive keyword line or LLM via `TITLE_MODE`). Child vectors embed **`retrieval_title + child text`**
- Embeds with SentenceTransformers (`all-MiniLM-L6-v2`), builds FAISS via `utils/rag_utils.py` / `utils/rag_index_build.py`
- Persists via `utils/rag_utils.save_vector_store` to `RAG_docs/vector_store/` and `utils/rag_index_build.save_parent_store` → `rag_parents.json`

**Prerequisites**: Knowledge files in `RAG_docs/knowledge/` (representative: MITRE ATT&CK, NIST SP 800-53, CIS Controls v8, ISO/IEC 27001)

**Run** (whenever knowledge files change):
```bash
jupyter notebook Index.ipynb
```

---

### 3b. Plan.ipynb (Action planning)

**Purpose**: Generate constrained mitigation plans from SHAP predictions, and run the **six-mode policy-corpus retrieval / reranking ablation** for `Score.ipynb`.

**What it does**:
- Loads the saved FAISS index and parent store from Part 1 (does not re-read `RAG_docs/knowledge/`)
- Builds retrieval queries per sample from the predicted label, confidence, party contributions, and top SHAP indicators (deterministic template, deterministic rephrase, optional LLM variants)
- For every sample, assembles context under the six modes in the [policy-corpus table](#policy-corpus-modes-this-repository)
- Expands the top-10 child chunks to full parent sections via `rag_parents.json` and calls OpenAI once per mode per sample
- Persists each sample’s full comparison (top-10 + 100-chunk candidate pool + LLM output for every mode) via `utils.rag_utils.save_rerank_comparison_report`
- Caches the doc2query-expanded corpus to `RAG_docs/rag_parents_doc2query.json`

**Prerequisites**:
- Part 1 completed (`RAG_docs/vector_store/` exists)
- Predictions from `Predict.ipynb`
- OpenAI API key in `.env`
- `RAG_docs/attack_options.json`, `RAG_docs/agentic_features.json`

**Inputs**:
- `RAG_docs/predictions/*.json` — predictions with SHAP
- `RAG_docs/vector_store/` — FAISS index + `rag_parents.json` from Part 1

**Outputs**:
- `RAG_docs/action_plans/rerank_comparison_sample_<id>_<ts>.json` — one per sample, all six modes
- `RAG_docs/rag_parents_doc2query.json` — cached doc2query-expanded corpus

**Run**:
```bash
jupyter notebook Plan.ipynb
```

---

### 4. Score.ipynb (Evaluation)

**Purpose**: Evaluate the six policy-corpus modes from `Plan.ipynb` with BEIR / CRAG-aligned **headline MiniLM metrics**, plus per-attack-class tables and grouped bar charts.

**What it does**:
- Loads every `rerank_comparison_sample_*.json` produced by `Plan.ipynb`
- Builds a deterministic **label-derived reference string** per sample from `attack_options.json` and `agentic_features.json`
- Derives **adaptive thresholds** per batch (median / 25th-percentile of the observed score distribution, clamped to relaxed floors)
- Computes the headline metrics per sample per mode and aggregates by rerank mode and by attack class
- Exports `RAG_docs/scoring/scoring_results.json` and grouped bar charts

**Prerequisites**: Rerank comparison reports from `Plan.ipynb`

**Inputs**:
- `RAG_docs/action_plans/rerank_comparison_sample_*.json`
- `RAG_docs/attack_options.json`
- `RAG_docs/agentic_features.json`

**Outputs**:
- `RAG_docs/scoring/scoring_results.json`
- Per-rerank-mode and per-attack-class tables
- Grouped bar charts (one group per metric)

**Execution Time**: 2–10 minutes (depending on number of samples)

**Run**:
```bash
jupyter notebook Score.ipynb
```

## Technologies & Tools

### Core ML/DL Framework
- **PyTorch** (≥2.0.0) — deep learning
- **NumPy** (≥1.23.0) — numerical computing
- **Pandas** (≥1.5.0) — data manipulation
- **scikit-learn** (≥1.2.0) — ML utilities

### Explainability
- **SHAP** (≥0.42.0) — SHapley Additive exPlanations

### LLM & RAG
- **OpenAI API** — constrained action-plan generation
- **LangChain** (≥0.3.0) — LLM application framework
- **FAISS** — dense vector similarity search (Stage-1 for dense modes)
- **Sentence Transformers** (≥2.2.0) — MiniLM bi-encoder + CrossEncoder reranker
- **rank-bm25** — lexical Stage-1 retriever (`bm25`, `bm25_crossencoder`)
- **Hugging Face Transformers** + **SentencePiece** — T5 (`castorini/doc2query-t5-base-msmarco`) for `doct5query`
- **RAGatouille** — ColBERT v2 late-interaction reranker (patched CPU MaxSim kernel on Windows)

### Evaluation
- **MiniLM (`all-MiniLM-L6-v2`)** — headline SBERT cosine / graded relevance
- **BERTScore** (≥0.3.13) — contextual similarity of LLM reasoning vs. reference
- **ROUGE** (≥1.0.1) — lexical overlap inside graded relevance
- **NLTK** (≥3.8.0) — tokenization
- **CRAG-style proxy** — 3-valued truthfulness rescaled to `[0, 1]` with adaptive per-batch thresholds
- **CySecBERT, CTI-BERT, SecureBERT** — parallel cyber-encoder graders in the full CyberRerankBench protocol

### Data Processing
- **pypdf** — PDF ingest for the policy knowledge base

### Development
- **Jupyter Notebook** — interactive pipeline
- **Python 3.8+** (3.12 recommended)

## Dataset

### UM-NIDS Dataset (Unified Multimodal Network Intrusion Detection Systems Dataset)

**Source**: [IEEE DataPort - UM-NIDS Dataset](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset)

The Unified Multimodal Network Intrusion Detection System (UM-NIDS) dataset integrates network flow data from CIC-IDS 2017, CIC-IoT 2023, UNSW-NB15, and CIC-DDoS 2019. This project uses the **undersampled CIC-IDS 2017** component.

**Dataset File Used**: `undersampled_CIC2017_dataset.csv`

**Key Statistics**:
- **Original Features**: 120 features (from CIC-IDS 2017)
- **Processed Features**: 97 features in the current VFL split (23 + 32 + 42 across three parties)
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

## Results

### Detection (VFL)

| Model | Accuracy | Macro Recall | Macro F1 |
|-------|----------|--------------|----------|
| **VFL** | 0.9617 | 0.9864 | 0.8594 |
| **Standard NN** | 0.9643 | 0.9863 | 0.8734 |

VFL matches centralized accuracy while keeping raw flows local to each party.

### Agent contributions (SHAP)

- **RAN**: 32.14% mean contribution
- **Edge**: 39.25% mean contribution
- **Core**: 28.61% mean contribution

These attributions are injected into RAG queries so retrieval is conditioned on *which domain evidence drove the detection*, not only the predicted class.

### Reranking (CyberRerankBench)

See [Findings](#findings). Dense MiniLM remains strongest on policy-corpus faithfulness; playbook retrieval lifts pool recall at the cost of utilization; cyber-encoder graders disagree with MiniLM and with each other. No mode wins uniformly across classes or encoders.

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure you're in project root
cd /path/to/VFL-RAG

# Verify utils directory exists
# Reinstall if needed
pip install -r requirements.txt
```

**CUDA/GPU Issues**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
# If False, reinstall PyTorch with CUDA support (see Installation)
```

**Dataset Not Found**:
- Verify CSV files are in `datasets/`
- Check file names (should be `.csv`)
- Run notebooks from the project root

**OpenAI API Errors**:
- Verify `.env` exists in the project root
- Check API key format: `OPENAI_API_KEY=sk-...`
- Ensure no extra spaces or quotes
- Test: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OK' if os.getenv('OPENAI_API_KEY') else 'Missing')"`

**Memory Issues**:
- Reduce batch size in the training notebook
- Use a smaller dataset subset for testing
- Close other applications
- Consider CPU if GPU memory is limited

**NLTK Data Missing**:
```python
import nltk
nltk.download('punkt')
```

**ColBERT on Windows**:
- Use Python 3.12 (RAGatouille/`voyager` has no cp313 wheels)
- First ColBERT load JIT-builds a torch extension; install MSVC “Desktop development with C++” and keep `ninja` on PATH

## Contributing

Contributions are welcome. Please follow these guidelines.

### How to Contribute

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/VFL-RAG.git
   cd VFL-RAG
   git remote add upstream https://github.com/original-owner/VFL-RAG.git
   ```

2. **Create Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**:
   - Follow PEP 8
   - Add type hints to functions
   - Include Google-style docstrings
   - Update documentation as needed

4. **Test Changes**:
   ```bash
   python -c "from utils.model_utils import VFLModel; print('Imports OK')"
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
- **Docstrings**: Google-style
- **Formatting**: Use `black`

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

## License

This project is licensed under the MIT License.

## Acknowledgments

- **UM-NIDS Dataset**: [IEEE DataPort](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset) (doi:10.21227/d8at-gb29)
- **CICIDS-2017**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **SHAP**: [SHapley Additive exPlanations](https://github.com/slundberg/shap)
- **LangChain**: [LangChain Framework](https://www.langchain.com/)
- **PyTorch**: [PyTorch](https://pytorch.org/)
- Policy corpora: MITRE ATT&CK, NIST SP 800-53, CIS Controls v8, ISO/IEC 27001 (and related mappings in `RAG_docs/knowledge/`)

---

This project is for research and educational purposes. Handle telemetry and policy documents in line with applicable privacy and licensing constraints.
