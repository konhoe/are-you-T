# CAS2105 HW6 Mini AI Pipeline — MBTI Post Classification (F vs T)

This repo contains a mini AI pipeline project for classifying MBTI forum post snippets into **Feeling (F)** vs **Thinking (T)**.

## Task
- **Input:** English post snippet (text)
- **Output:** Label in {F, T}
- **Metrics:** Accuracy, Macro-F1

## Dataset
- **Source:** Kaggle Myers-Briggs Personality Type Dataset (MBTI)
- Each row contains an MBTI type (e.g., INFJ) and a long `posts` field.
- The label is derived from the **3rd letter** of the MBTI type: `F` or `T`.
- The `posts` field is split by `|||` into many snippets, so the total number of samples becomes large.

## Methods

### 1) Naive baseline (keyword rule)
Two keyword sets are used:

- FEEL_WORDS = {feel, feeling, emotion, love, like, heart, emotional, care, compassion, empathy, relationship}
- THINK_WORDS = {logic, reason, analyze, analysis, objective, data, rational, facts, structure, system, argument}

Rule:
- Count occurrences (substring match) in the lowercased text.
- If FEEL count > THINK count → predict **F**
- Else (including tie) → predict **T**

### 2) AI pipeline (DeBERTa encoder + MLP head)
- Encoder: `microsoft/deberta-v3-base`
- Classifier head: small MLP (Linear → ReLU → Dropout(0.1) → Linear(2))
- Training: full fine-tuning (encoder + head)

## Results (from logs)

### Main dataset (after splitting)
- **Baseline Test:** acc=0.5181, macro_f1=0.5079 (n=76,670)
- **AI Pipeline Test:** acc=0.6500, macro_f1=0.6445 (pred_dist=[22351, 15984])

### Small synthetic set (n=100)
- **Baseline inference:** acc=0.7200, macro_f1=0.7190
- **AI inference:** acc=0.6600, macro_f1=0.6511

## How to run

### Baseline
```bash
python naive_method/baseline.py
```

### Train AI model
```bash
python src/train.py
```

### Inference model
```bash
python src/inference.py
python naive_method/baselin_inf.py
```
