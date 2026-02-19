# CLARION-Lite: Biomedical Document Summarization

This repository contains a lightweight hierarchical inference framework for long biomedical text summarization using a pre-trained BART-large model.

## 📌 Overview

Long biomedical research articles often exceed transformer input limits. CLARION-Lite uses a chunking + refinement strategy that:

1. Splits long documents into chunks
2. Summarizes each chunk independently
3. Merges intermediate summaries
4. Produces a final concise summary

## 📂 Repository Structure

