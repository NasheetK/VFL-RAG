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

### Key Innovation

- **Next-Gen 5G Networking**: Agentic AI system designed for distributed 5G network architectures
- **Vertical Feature Partitioning**: Features distributed across RAN, Edge, and Core agents based on 5G network observability
- **Privacy-Preserving Federated Learning**: Collaborative learning without sharing raw data across network tiers
- **SHAP-Based Explainability**: Agent-level attribution scores guide automated mitigation recommendations  
- **Agentic Reasoning**: Intelligent LLM-powered action planning using RAG (Retrieval-Augmented Generation)
- **High Performance**: >96% test accuracy on standardized intrusion detection datasets

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

**UM-NIDS Dataset (Unified Multimodal Network Intrusion Detection Systems Dataset)**:
- **Source**: [IEEE DataPort - UM-NIDS Dataset](https://ieee-dataport.org/documents/unified-multimodal-network-intrusion-detection-systems-dataset)
- **Description**: Comprehensive, standardized dataset integrating network flow data from multiple well-established datasets including CIC-IDS 2017
- **Dataset Used**: Undersampled version of CIC-IDS 2017 dataset
- **File**: `undersampled_CIC2017_dataset.csv`
- **Processing**: 88 features (from original 120, 32 removed for 5G observability)
- **Size**: ~390,000 labeled flow records
- **Classes**: 9 attack types (BENIGN, BOT, DDOS, DOS, FTPPATATOR, OTHERS, PORTSCAN, SSHPATATOR, WEBATTACK)

**Setup**:
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

**Note**: 
- IEEE DataPort subscription required for dataset access
- The project uses the undersampled version for faster experimentation
- For full evaluation, use the complete standardized CIC-IDS 2017 dataset from UM-NIDS

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
python -c "from utils.model import VFLModel; from utils.vfl_utils import load_agent_definitions; print('✓ Setup OK')"

# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📓 Notebooks Workflow

Execute notebooks in this order:

```
VFL_SHAP_MultiClass.ipynb  →  VFL_SHAP_Prediction.ipynb  →  RAG_LLM_action_plan.ipynb  →  scoring_evaluation.ipynb
```

### 1. VFL_SHAP_MultiClass.ipynb (Training)

**Purpose**: Train VFL and centralized baseline models

**Why**: Builds the core intrusion detection models with privacy-preserving federated learning

**What it does**:
- Loads `undersampled_CIC2017_dataset.csv` from `datasets/` directory
- Partitions features into RAN (23), Edge (32), Core (42) agents
- Trains VFL model with vertical federated learning
- Trains centralized baseline for comparison
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
jupyter notebook VFL_SHAP_MultiClass.ipynb
```

---

### 2. VFL_SHAP_Prediction.ipynb (Inference)

**Purpose**: Generate predictions for new network flows

**Why**: Apply trained models to new data and get explainable predictions

**What it does**:
- Loads trained models from `model/` directory
- Processes new network flow data
- Generates predictions with confidence scores
- Computes SHAP values for each prediction
- Identifies dominant agent per prediction

**Prerequisites**: Must run `VFL_SHAP_MultiClass.ipynb` first

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
jupyter notebook VFL_SHAP_Prediction.ipynb
```

---

### 3. RAG_LLM_action_plan.ipynb (Action Planning)

**Purpose**: Generate context-aware mitigation action plans

**Why**: Converts SHAP attributions into actionable security recommendations

**What it does**:
- Loads predictions with SHAP attributions
- Builds/loads FAISS vector store from knowledge base PDFs
- Performs semantic search using RAG
- Generates action plans using OpenAI GPT models
- Incorporates agent-specific mitigation strategies

**Prerequisites**: 
- Predictions from `VFL_SHAP_Prediction.ipynb`
- OpenAI API key in `.env` file
- Knowledge base PDFs in `RAG_docs/knowledge/`

**Inputs**:
- `RAG_docs/predictions/predictions_*.json` - Predictions with SHAP
- `RAG_docs/agentic_features.json` - Agent definitions
- `RAG_docs/attack_options.json` - Attack mitigation options
- PDF files in `RAG_docs/knowledge/` - Knowledge base

**Outputs**: 
- `RAG_docs/vector_store/` - FAISS vector store (created on first run)
- `RAG_docs/action_plans/action_plan_*.json` - Generated action plans

**Execution Time**: 5-15 minutes (first run builds vector store)

**Configuration**:
```python
model_name = "gpt-4"  # or "gpt-3.5-turbo"
temperature = 0.7
max_tokens = 1000
top_k = 5  # Number of documents to retrieve
```

**Run**:
```bash
jupyter notebook RAG_LLM_action_plan.ipynb
```

---

### 4. scoring_evaluation.ipynb (Evaluation)

**Purpose**: Evaluate action plan quality using NLP metrics

**Why**: Validates that generated recommendations are accurate and contextually appropriate

**What it does**:
- Loads generated action plans
- Computes ROUGE-1 (lexical precision) - word-for-word overlap
- Computes SBERT Cosine similarity (semantic alignment) - intent matching
- Computes BERTScore F1 (reasoning depth) - contextual awareness
- Generates evaluation reports and visualizations

**Prerequisites**: Action plans from `RAG_LLM_action_plan.ipynb`

**Inputs**:
- `RAG_docs/action_plans/action_plan_*.json` - Generated action plans
- `RAG_docs/attack_options.json` - Reference actions
- `RAG_docs/agentic_features.json` - Agent definitions

**Outputs**: 
- `RAG_docs/scoring/scoring_results.json` - Evaluation results
- Console output with metric scores
- Visualizations (if matplotlib configured)

**Execution Time**: 2-10 minutes (depending on number of action plans)

**Metrics**:
1. **ROUGE-1**: Lexical precision (word overlap) - ensures compliance with allowed actions
2. **SBERT Cosine**: Semantic similarity (0-1 scale) - measures strategic alignment
3. **BERTScore F1**: Contextual reasoning (0-1 scale) - validates evidence mapping

**Run**:
```bash
jupyter notebook scoring_evaluation.ipynb
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
- **OpenAI API** - GPT-4/GPT-3.5 for action plan generation
- **LangChain** (≥0.3.0) - LLM application framework
- **FAISS** - Vector similarity search for RAG
- **Sentence Transformers** (≥2.2.0) - Semantic embeddings

### Evaluation
- **BERTScore** (≥0.3.13) - Contextual similarity evaluation
- **ROUGE** (≥1.0.1) - Lexical overlap metrics
- **NLTK** (≥3.8.0) - Natural language processing

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
| **Centralized** | 0.9643 | 0.9863 | 0.8734 |

**Key Finding**: VFL achieves comparable performance (<0.3% difference) while maintaining data privacy.

### Agent Contributions (SHAP Analysis)

- **RAN**: 32.14% mean contribution
- **Edge**: 39.25% mean contribution  
- **Core**: 28.61% mean contribution

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
│   ├── model.py          # Model definitions
│   └── vfl_utils.py      # Utility functions
├── RAG_docs/             # RAG knowledge base
│   ├── agentic_features.json
│   ├── attack_options.json
│   ├── knowledge/        # PDF knowledge base
│   └── vector_store/     # FAISS vector store
├── outputs/              # Training outputs
├── inputs/               # Input data for prediction
├── VFL_SHAP_MultiClass.ipynb
├── VFL_SHAP_Prediction.ipynb
├── RAG_LLM_action_plan.ipynb
├── scoring_evaluation.ipynb
└── requirements.txt
```

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
   python -c "from utils.model import VFLModel; print('✓ Imports OK')"
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
