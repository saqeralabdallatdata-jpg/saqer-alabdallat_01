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
┌────────────────────────────────────────────────────────┐
│             Streamlit Candidate Analytics Dashboard    │
│            (Match Matrix, Gap Radar & Summary)         │
└──────────────────────────┴─────────────────────────────┘
