# Clarion Lite Biomedical Summarization

This repository contains the implementation of a hierarchical biomedical document summarization framework using a pre-trained BART-large model.

## 📌 Project Overview

Biomedical research articles are often lengthy and complex. This project proposes a chunking-based hierarchical summarization strategy to handle long documents efficiently.

## ⚙️ Methodology

1. Long biomedical document is divided into smaller chunks.
2. Each chunk is processed using BART-large.
3. Generated chunk summaries are merged.
4. Final concise summary is produced.

## 📊 Evaluation Metrics

ROUGE Scores:
- ROUGE-1: 0.2938
- ROUGE-2: 0.1001
- ROUGE-L: 0.1973

## 🧠 Model Used

- facebook/bart-large (HuggingFace Transformers)

## 📂 Repository Structure

- main.py – summarization pipeline
- plot.py – ROUGE visualization
- requirements.txt – dependencies

## 📜 Research Paper

This work is prepared for IEEE International Conference submission.

