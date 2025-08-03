# Probing vs Fine-Tuning: Benchmarking BERT on News Classification

**Authors:** Jack Parry-Wingfield  
**Model:** `bert-base-uncased`  
**Dataset:** AG News Topic Classification

## LaTeX Documentation

## Overview

This project explores how well frozen vs. fine-tuned BERT embeddings perform on a real-world text classification task. We benchmarked probing methods like Logistic Regression and K-Nearest Neighbors using sentence-level embeddings, and compared those results to end-to-end fine-tuning of the entire BERT model.

---

## Dataset

**AG News** is a 4-class topic classification dataset derived from over 1M news articles.  
Each sample includes a short **title + description** from one of the following categories:
- World
- Business
- Sports
- Sci/Tech

- 120,000 training samples  
- 7,600 test samples  
- Loaded via `datasets`  
- Tokenized with `bert-base-uncased` tokenizer (max length: 128)

---

## Methods

### Probing

- **Frozen BERT** used as a feature extractor
- Sentence representations:
  - `[CLS]` token
  - Mean pooling (best performance!)
  - Last token embedding
- Downstream classifiers:
  - Logistic Regression
  - KNN (K optimized via val set)
- Full dataset (120k) used for training

> **Best probing result:** 91.13% accuracy (Mean pooling + Logistic Regression)

### Fine-Tuning

- All BERT layers unfrozen
- Trained on a 60k subset (80/20 train/val split)
- Optimizer: AdamW  
- LR: 2e-5 | Epochs: 3 | Batch Size: 16  
- Model: `BertForSequenceClassification`

> **Final test accuracy:** 94.51%

### Attention Visualization

Used `bertviz` to inspect attention heads:
- Correct vs incorrect predictions
- Positive vs negative class predictions
- Focus patterns reveal tight attention on key terms in correct cases

---

## Results Summary

| Method                     | Accuracy (%) |
|---------------------------|--------------|
| Probing (Mean + LogReg)   | 91.13        |
| Fine-tuned BERT (60k)     | **94.51**    |

> Fine-tuning led to a +3.38% boost, but required more compute (Colab A100 used).

---
