# News Topic Intelligence & MLOps Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-DeBERTa%20%7C%20DistilBART-FFD21E)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow%20%26%20DVC-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI-Pytest%20%7C%20Ruff%20%7C%20Bandit-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A **production-grade multimodal news intelligence and MLOps platform** — featuring transformer-based topic classification, DistilBART abstractive summarization, multi-source RAG conversational intelligence, A/B testing infrastructure, Stable Diffusion thumbnail generation, and a **continuous Human-in-the-Loop (HITL) Active Learning loop**.

Built with end-to-end engineering discipline: reproducible pipelines via DVC and MLflow, hardened FastAPI microservice, automated Alembic migrations, Prometheus metrics, and a full Streamlit command center.

---

## 🏗️ System Architecture

<p align="center">
  <img src="docs/images/system_architecture.png" alt="News Topic Intelligence Platform Architecture" width="95%"/>
</p>

---

## ✨ Key Platform Capabilities

| Capability | Module | Technical Highlights |
|---|---|---|
| **Text & Topic Classification** | `app/services/classifier.py` | Fine-tuned DeBERTa transformer ensemble with TF-IDF baseline fallback and calibrated confidence scores |
| **Abstractive Summarization** | `app/services/summarizer.py` | DistilBART pipeline providing high-compression abstractive summaries with latency guarantees |
| **Multimodal Intelligence** | `image_generation/` | CLIP + BLIP vision-language joint embeddings for cross-modal article and image topic classification |
| **Multi-Source RAG Chatbot** | `app/services/chatbot_service.py` | Multi-turn conversational agent retrieving across 200K+ news corpus, platform docs, and live analytics with citations |
| **A/B Testing Infrastructure** | `app/services/ab_testing_service.py` | Hash-based deterministic user traffic splitting (Ensemble vs. Transformer) with automated metric tracking |
| **Active Learning Safety Net** | `app/services/active_learning.py` | Automatic queuing of low-confidence predictions into a review pool with human feedback ingestion |
| **AI Image Generation** | `scripts/simple_image_generator.py` | Stable Diffusion 1.5 thumbnail generation with GPU acceleration (RTX/CUDA optimized) |
| **Enterprise Hardening** | `app/main.py` | API key enforcement, structured JSON logging, Prometheus metrics (`/metrics`), request tracing (`x-request-id`) |
| **Full Streamlit Dashboard** | `dashboard/streamlit_app.py` | Stakeholder command center: classification, summarization, live trends, review queue, and chatbot UI |

---

## 🔄 Continuous Active Learning & Stream Review Loop

Low-confidence predictions from production are never lost. They are automatically diverted to a structured Human-in-the-Loop review queue to continuously improve model quality.

<p align="center">
  <img src="docs/images/active_learning_loop.png" alt="Active Learning & Stream Review Loop" width="95%"/>
</p>

```
1. Live Request (/classify_news) 
   ──► Score < threshold? ──► Auto-diverted to Review Queue (SQLite/PostgreSQL)
2. Auditor / Reviewer opens Streamlit Review Center 
   ──► Validates or re-labels true category
3. Feedback ingestion (scripts/manage.py)
   ──► Merges feedback into DVC dataset ──► Triggers MLflow fine-tuning run
```

---

## 🌐 API Reference

The FastAPI service exposes production-hardened REST endpoints with OpenAPI documentation at `/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Service readiness probe & dependency health status |
| `/classify_news` | `POST` | Single-article topic classification with confidence scoring |
| `/batch_classify` | `POST` | High-throughput batch classification with review threshold filtering |
| `/summarize_news` | `POST` | DistilBART abstractive article summarization |
| `/ab_classify` | `POST` | A/B testing prediction with user-variant traffic splitting |
| `/feedback` | `POST` | Ingest human reviewer label corrections into the active learning pool |
| `/chat` | `POST` | Multi-source RAG conversational query with source citations |
| `/metrics` | `GET` | Prometheus operational & latency metrics export |

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- Git
- (Optional) NVIDIA GPU with CUDA for local Stable Diffusion & transformer acceleration

### 1. Clone & Environment Setup

```powershell
git clone https://github.com/HayderCH/WATS---News-Transformer-Classification-service.git
cd WATS---News-Transformer-Classification-service

# Create virtual environment
python -m venv .venv
. .venv/Scripts/Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

```powershell
Copy-Item .env.example .env
# Edit .env to set your API_KEY and optional MLflow / S3 toggles
```

### 3. Initialize Database & Seed Content

```powershell
# Run database migrations
python -m alembic -c alembic.ini upgrade head

# Seed initial categories and review samples
python scripts/manage.py seed-db --overwrite
```

### 4. Start Services

```powershell
# Terminal 1: Launch FastAPI Backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Launch Streamlit Command Center
streamlit run dashboard/streamlit_app.py
```

The API docs will be live at `http://localhost:8000/docs` and the Dashboard at `http://localhost:8501`.

---

## 🐳 Docker Deployment

Run the complete platform (API + Database + Dashboard) in one command:

```bash
docker compose up --build
```

---

## 🛠️ CLI & MLOps Tooling (`scripts/manage.py`)

A comprehensive Typer CLI automates the entire machine learning lifecycle:

```powershell
# Train baseline TF-IDF model
python scripts/manage.py train-baseline --data-path data/raw/news.csv

# Fine-tune Transformer model with MLflow tracking
python scripts/manage.py train-transformer --epochs 3 --batch-size 16

# Evaluate models on test split
python scripts/manage.py evaluate --model-path artifacts/models/deberta-v3

# Ingest active learning feedback into training dataset
python scripts/manage.py merge-feedback --output-path data/processed/retrain.csv

# Package and bundle artifacts for deployment
python scripts/manage.py bundle-artifacts --output-dir artifacts/bundles
```

---

## 🧪 Testing & CI/CD Pipeline

The project enforces strict software engineering standards with a multi-stage GitHub Actions CI pipeline:

```powershell
# Run unit & integration tests (20 test suites)
pytest tests/ -v

# Run linting & formatting checks
ruff check .

# Run security static analysis
bandit -r app/ -ll

# Run dependency vulnerability audit
pip-audit
```

---

## 📁 Repository Structure

```text
├── .github/workflows/          CI/CD: Pytest, Ruff, Bandit, Retraining pipelines
├── alembic/                    SQLAlchemy database migration scripts
├── app/                        FastAPI application core
│   ├── api/                    REST route handlers (/classify, /summarize, /chat, etc.)
│   ├── core/                   Settings, security, authentication, structured logging
│   ├── db/                     SQLAlchemy database models & session management
│   ├── models/                 Pydantic request/response schemas
│   └── services/               Classifier, Summarizer, RAG Chatbot, A/B Testing
├── artifacts/                  Model checkpoints, tokenizers, feature importance metrics
├── dashboard/                  Streamlit Command Center (interactive analytics UI)
├── data/                       DVC metadata, raw datasets, category taxonomies
├── docs/                       Architecture documentation, runbooks, feature specs
├── image_generation/           CLIP/BLIP multimodal and Stable Diffusion modules
├── notebooks/                  Exploratory data analysis & model prototyping notebooks
├── scripts/                    Typer CLI (manage.py), setup scripts, image generators
├── tests/                      20 unit and integration test suites
├── Dockerfile                  Container build definition
├── docker-compose.yml          Multi-container composition
└── requirements.txt            Locked production dependencies
```

---

## 📖 Additional Documentation

- [Project Specification](docs/PROJECT_SPEC.md)
- [A/B Testing Methodology](docs/AB_TESTING.md)
- [Active Learning & Stream Review Guide](docs/STREAM_REVIEW.md)
- [Multi-Source RAG Chatbot Roadmap](docs/CHATBOT_ROADMAP.md)
- [Production Operations Runbook](docs/RUNBOOK.md)
- [Security & Threat Model](docs/SECURITY.md)

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
