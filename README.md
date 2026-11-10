# Veritas AI: Multimodal Content Verification System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-6.0+-646CFF.svg)](https://vitejs.dev)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC.svg)](https://tailwindcss.com)

Veritas AI is an academic multimodal content verification system developed for university Natural Language Processing (NLP) research. The system analyzes paired text-image claims and classifies digital submissions into **REAL**, **FAKE**, or **UNCERTAIN**.

---

## Key Features

- **Multimodal Neural Backbones**: DistilBERT (`distilbert-base-uncased`) for text and ResNet50 (`torchvision.models.resnet50`) for images.
- **Multimodal Fusion Architectures**: Comparative evaluation of **Feature-Level Fusion** (projected concatenation) and **Decision-Level Fusion** (out-of-fold trained meta-classifier).
- **Dual Modality Explainability**:
  - Text: Local Shapley attributions via `shap.Explainer` (partition algorithm, evaluation budget 100) with asynchronous polling and SHA-256 caching.
  - Image: Activation saliency via Grad-CAM on ResNet50 `layer4[-1]`.
  - Modality Attribution: Sensitivity drop ($\Delta P$) on modality ablation.
- **Selective Classification**: Post-hoc validation temperature scaling (minimizing NLL) and empirical thresholding ($\tau^*$) routing ambiguous or low-confidence claims to `UNCERTAIN`.
- **Scholarly Interface**: A clean, light academic journal UI designed for readability, transparency, and auditability.

---

## Repository Structure

```
Veritas-AI/
├── backend/          # Python FastAPI application, SQLite persistence, PyTorch serving
├── frontend/         # React + Vite + TypeScript + Tailwind CSS client
├── training/         # Google Colab training scripts and notebooks
├── data/             # Benchmark data (fakeddit) and isolated demo data (demo_curated)
├── docs/             # Research paper drafts and model cards
├── PROJECT_CONTEXT.md# Engineering specifications and constraints
└── README.md
```

---

## Quickstart

### 1. Backend Setup
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python run.py
```
Backend API will be live at `http://127.0.0.1:8000`. Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend development server will be live at `http://localhost:5173`.

---

## License & Attribution
- Fakeddit dataset: Nakamura et al., 2020.
- Demo imagery: CC0 Public Domain or Creative Commons licenses (see `data/demo_curated/demo_manifest.json`).
