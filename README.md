# Medical Text Classification with Explainable AI (XAI)

This project benchmarks classic classifiers on multiple text representations for **medical abstract classification**, and layers **explainability** on top (global + local) to make model decisions auditable.

We evaluate **Naïve Bayes, Decision Tree, Logistic Regression,** and **linear SVM** across **TF-IDF, Word2Vec, GloVe, SBERT,** and **BioBERT**, then analyze model behavior with **LIME**, **SHAP**, and **Integrated Gradients**.

---

## 🗂 Dataset

- **Medical-Abstracts-TC-Corpus** — added as a **Git submodule**:
  - Upstream: https://github.com/sebischair/Medical-Abstracts-TC-Corpus
  - Clone this repo **with submodules**:
    ```bash
    git clone --recurse-submodules https://github.com/Ay0u8/Medical-text-classification-XAI-.git
    cd Medical-text-classification-XAI-
    git submodule update --init --recursive
    ```

- Protocol (as used in the study):
  - 5 target classes
  - 80/20 train/validation split
  - Stratified **5-fold CV** on the training set for model selection
  - Metrics: **Accuracy, Precision, Recall, Macro-F1** (macro-F1 emphasized)

---

## 🧠 Methods

### Text Representations
- **TF-IDF** (sparse, interpretable)
- **Word2Vec / GloVe** (dense, context-independent; doc = mean of word vectors)
- **SBERT / BioBERT** (contextual embeddings; no fine-tuning in this study)

### Classifiers
- **Multinomial NB** (TF-IDF) / **Gaussian NB** (dense)
- **Linear SVM**, **Logistic Regression**
- **Pruned Decision Tree** (transparent baseline)

### Explainability (XAI)
- **Global**: NB class-top tokens; LR/SVM coefficients; tree paths/feature importances
- **Local**: **LIME** token attributions, **SHAP** (KernelSHAP/TreeSHAP), **Integrated Gradients** for transformer embeddings
- **Faithfulness** checks via perturbation/deletion-insertion style curves

---

## 🔎 Key Findings (high level)

- **SBERT** and especially **BioBERT** paired with **Linear SVM / Logistic Regression** outperform TF-IDF/Word2Vec/GloVe.
- **TF-IDF + SVM** provides a strong, transparent baseline.
- **Decision Trees** underperform on dense embedding spaces but remain useful for interpretability demos.

> See the notebook / HTML export for exact per-model CV scores and plots.

---

## 📂 Project Structure

- `Medical-Abstracts-TC-Corpus/` – dataset (Git submodule)  
- `XAI.ipynb` – main experiments (training + XAI)  
- `BSP4.pdf` – full project report  
- `figures/` – saved plots (global + local XAI)  
- `requirements.txt` – project dependencies  
- `.gitmodules` – submodule configuration  
- `.gitignore` – ignored files and folders

---

## 📊 Results (high level)

- **BioBERT + Linear SVM** → best overall performance.
- **SBERT + (SVM or Logistic Regression)** → strong runner-up.
- **TF-IDF + SVM** → robust, transparent baseline.
- **Word2Vec ≥ GloVe** in this setup.
- **Decision Trees** underperform on dense embeddings but remain useful for interpretability.

---

## 🧪 Reproducibility

- **Data splits:** 80/20 train/validation with **stratified 5-fold CV** on training.
- **Random seeds:** fixed for dataset splits, model training, and sampling.
- **Preprocessing:** deterministic tokenization/normalization; saved vectorizers/encoders.
- **Class imbalance:** class weights where supported (or document in notebook).
- **Tracking:** store metrics (Accuracy, Precision, Recall, Macro-F1) per fold/run.
- **Artifacts:** save trained models (when small), vectorizers, and XAI outputs to `figures/`.
- **Environment:** list pinned versions in `requirements.txt` (Python 3.10+).
- **Re-run:** notebook cells execute end-to-end without manual edits; paths are relative.





