# 🎯 Smart Talent ATS & Resume Matcher Engine (v2.0)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-LLM_Ops-green?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.25+-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Production-grade Applicant Tracking System (ATS) leveraging Large Language Models, contextual vector embeddings, and structured JSON parsing to evaluate candidate-job fit with high precision.**

---

## 📌 Problem Statement & Core Value Proposition

Traditional Applicant Tracking Systems rely on rigid keyword matching (TF-IDF/Exact Phrase) which often disqualifies qualified candidates who use alternative phrasing or syntax. 

**Smart Talent ATS Engine** overcomes these limitations by combining **Semantic Vector Similarity** with **LLM Reasoning**:
* **Multi-Format Parsing:** Extracts text cleanly from PDF/DOCX layouts, handling multi-column tables and complex headers.
* **Semantic Embedding Matcher:** Measures cosine similarity using vector representations rather than simple strings.
* **Structured Gap Analysis:** Automatically identifies missing technical skills, domain experiences, and certifications.
* **Deterministic Output Schema:** Uses Pydantic JSON parsing to guarantee valid API outputs for downstream enterprise integrations.

---

## 🏗️ System Architecture

```text
[ Resume PDF / DOCX ]         [ Job Description (JD) ]
          │                              │
          ▼                              ▼
┌────────────────────────────────────────────────────────┐
│               Document Parsing & Cleaning              │
│       (PyPDF / pdfplumber / Regex Normalization)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             LangChain & LLM Extraction Engine          │
├──────────────────────────┬─────────────────────────────┤
│   Semantic Vector Match  │   Structured Gap Inference  │
│  (Cosine Similarity)     │    (Pydantic Output Guard)  │
└──────────────────────────┴─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI REST Middleware                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼


🛠️ Tech Stack & Dependencies
Core Language: Python 3.10+

LLM & Embeddings Framework: LangChain / LangGraph, OpenAI / HuggingFace Embeddings

Backend API Layer: FastAPI, Uvicorn, Pydantic

Frontend Analytics Console: Streamlit, Plotly

Data Processing & Utilities: PyPDF2, pdfplumber, Scikit-learn, NumPy

🚀 Quick Start Guide
1. Clone & Setup Environment

git clone [https://github.com/saqer-alabdallat/smart-ats-resume-engine.git](https://github.com/saqer-alabdallat/smart-ats-resume-engine.git)
cd smart-ats-resume-engine

2. Install Dependencies

pip install -r requirements.txt

3. Configure Environment Variables
Create a .env file in the root directory:

4. Run Application

# Terminal 1: Launch API Core
uvicorn backend.main:app --reload --port 8000

# Terminal 2: Launch Streamlit Dashboard
streamlit run frontend/app.py
