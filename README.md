# Teen Stress Level Prediction

ML Zoomcamp Midterm Project: Predicting teenager self-reported stress levels (1–5) based on lifestyle, demographic, and behavioural factors.

---

## 📌 Project Overview

Adolescence is a critical developmental stage where stress can significantly impact mental health and well-being. This project aims to predict a teenager's stress level (on a 1–5 scale) using daily habits such as sleep duration, screen time, physical exercise, and social media usage.

---

## 📁 Project Structure

```text
teen-stress-prediction/
├── app/                  # Streamlit web application
├── data/                 # Raw and processed datasets
│   └── mental_health_analysis.csv
├── models/               # Serialized trained models and pipelines
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── src/                  # Reusable Python modules (data processing, training, inference)
├── Dockerfile            # Container configuration
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Deterministic dependency lock file
├── README.md             # Project documentation
```

---

## 🛠️ Environment Setup & Installation

This project uses [`uv`](https://docs.astral.sh/uv/) for Python package and environment management.

### 1. Install `uv` (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Project Dependencies

```bash
uv sync
```

This creates `.venv/` automatically. Run commands with `uv run`, or activate manually:

```bash
source .venv/bin/activate
```

### 3. Run Training, Prediction, App

```bash
uv run python src/train.py
uv run python src/predict.py
uv run streamlit run app/app.py
```

---

## 📊 Dataset

The dataset is sourced from Kaggle ([Mental Health Analysis Among Teenagers](https://www.kaggle.com/datasets/aniruddhawankhede/mental-heath-analysis-among-teenagers)).

### Key Features:
- `Age`: Age of the participant (13–19 years)
- `Gender`: Gender (`M` / `F`)
- `Social_Media_Hours`: Daily time spent on social media (hours)
- `Exercise_Hours`: Daily physical activity / exercise (hours)
- `Sleep_Hours`: Daily sleep duration (hours)
- `Screen_Time_Hours`: Total daily screen time (hours)
- `Support_System`: Level of support system (`Low`, `Moderate`, `High`)
- `Academic_Performance`: Academic performance rating (`Poor`, `Average`, `Good`, `Excellent`)
- `Survey_Stress_Score`: Self-reported stress level (1–5) (**Target**)
- `Wearable_Stress_Score`: Objective stress score (0–1) from wearable device (*excluded during modeling to prevent leakage*)

---

## 🚀 Workflow Roadmap

1. [x] **Project setup** (Environment, dependencies, dataset, folder structure)
2. [ ] **Exploratory Data Analysis (EDA)** (`notebooks/01_eda.ipynb`)
3. [ ] **Problem Definition & Validation Strategy**
4. [ ] **Baseline Model** (Logistic Regression)
5. [ ] **Model Experiments & Comparison** (Decision Trees, Random Forest, XGBoost)
6. [ ] **Hyperparameter Tuning**
7. [ ] **Model Analysis & Explainability** (Feature importance, Confusion matrix)
8. [ ] **Reproducible Training Pipeline** (`src/train.py`, `src/predict.py`)
9. [ ] **Interactive Streamlit UI** (`app/app.py`)
10. [ ] **Containerization with Docker**
