# Teen Stress Level Prediction

ML Zoomcamp Midterm Project: Predicting teenager self-reported stress levels (1-5) from lifestyle, demographic, and behavioural factors.

**Live demo:** [mlzoomcampmidterm-ngv5vx8ao6wqc3gw5tpsyy.streamlit.app](https://mlzoomcampmidterm-ngv5vx8ao6wqc3gw5tpsyy.streamlit.app/)

---

## Project Overview

Adolescence is a critical developmental stage where stress can significantly affect mental health and well-being. This project estimates a teenager's self-reported stress level (1-5) from daily habits such as sleep duration, screen time, physical exercise, social media use, plus basic demographic and academic context.

The deployed Streamlit app is an informational tool: a teenager, parent, or educator moves the sliders and immediately sees the estimated stress level and the model's probability for each level. It is **not** a diagnostic or medical tool.

This is a hard, low-signal prediction task: in this dataset the daily-habit features are almost uncorrelated with the self-reported stress score, so a realistic model performs only slightly better than random guessing. Documenting that honestly is part of the result.

---

## Dataset

The dataset is committed in this repository at `data/mental_health_analysis.csv` (5,000 rows, 11 columns, no missing values), so the project can be reproduced without downloading anything.

Original source: [Mental Health Analysis Among Teenagers on Kaggle](https://www.kaggle.com/datasets/aniruddhawankhede/mental-heath-analysis-among-teenagers).

### Key Features

- `Age`: Age of the participant (13-18 years)
- `Gender`: Gender (`M` / `F`)
- `Social_Media_Hours`: Daily time spent on social media (hours)
- `Exercise_Hours`: Daily physical activity / exercise (hours)
- `Sleep_Hours`: Daily sleep duration (hours)
- `Screen_Time_Hours`: Total daily screen time (hours)
- `Support_System`: Level of support system (`Low`, `Moderate`, `High`)
- `Academic_Performance`: Academic performance rating (`Poor`, `Average`, `Good`, `Excellent`)
- `Survey_Stress_Score`: Self-reported stress level (1-5), the **target**
- `Wearable_Stress_Score`: Objective stress score (0-1) from a wearable device, **excluded during modeling to prevent target leakage**
- `User_ID`: Identifier, dropped before modeling

---

## Models and Results

The notebook compares several models and tunes the tree-based one:

1. **Logistic Regression** (scaled numeric features + one-hot encoded categorical features)
2. **Decision Tree**
3. **Random Forest** tuned with `GridSearchCV` over `n_estimators`, `max_depth`, and `min_samples_split`
4. **Final Logistic Regression** refit on the full training set

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| Final Logistic Regression | 0.215 | 0.211 |
| Logistic Regression baseline | 0.209 | 0.204 |
| Decision Tree | 0.194 | 0.194 |
| Tuned Random Forest (validation) | - | 0.204 (CV) |

With five balanced classes, random guessing gives a macro F1 of about 0.20, so the final model is only marginally better than chance. Permutation importance is near zero for every feature. This is a property of the dataset (near-zero correlations between habits and self-reported stress), not a bug in the pipeline.

---

## Project Structure

```text
ml_zoomcamp_midterm/
├── app/
│   └── app.py                     # Streamlit web application
├── data/
│   └── mental_health_analysis.csv # Committed dataset (5,000 rows)
├── models/
│   ├── final_pipeline.joblib      # Serialized preprocessing + model
│   └── final_metadata.joblib      # Saved test metrics
├── notebooks/
│   ├── notebook_main.ipynb        # Clean end-to-end notebook (EDA -> model -> save)
│   └── training/                  # Step-by-step learning notebooks
├── src/
│   ├── train.py                   # Trains the final model and saves it
│   └── predict.py                 # Loads the model and scores one profile
├── Dockerfile                     # Container for the Streamlit service
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Deterministic dependency lock file
├── requirements.txt               # Runtime deps used by Streamlit Community Cloud
└── README.md
```

---

## Environment Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for Python package and environment management (Python 3.12+).

### 1. Install `uv` (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the environment and install dependencies

```bash
uv sync
```

This creates `.venv/` automatically and installs the exact versions from `uv.lock`.

### 3. Activate the environment

```bash
source .venv/bin/activate
```

Alternatively, prefix every command with `uv run` and skip activation.

---

## Running the Project

```bash
# 1. Train the final model and write models/final_pipeline.joblib
uv run python src/train.py

# 2. Score a sample teenager from the command line
uv run python src/predict.py

# 3. Launch the interactive app
uv run streamlit run app/app.py
```

Then open <http://localhost:8501>.

The training script prints the train/validation/test sizes plus test accuracy and macro F1, and saves the fitted pipeline and metrics under `models/`.

---

## Docker

Build the image from the repository root:

```bash
docker build -t teen-stress-prediction .
```

Run the container, mapping Streamlit's port to the host:

```bash
docker run --rm -p 8501:8501 teen-stress-prediction
```

Open <http://localhost:8501>. The image installs dependencies with `uv sync --frozen` from `pyproject.toml` + `uv.lock`, copies `src/`, `models/`, and `app/`, and starts Streamlit on `0.0.0.0:8501`. A health check is available at `/_stcore/health`.

---

## Deployment

The service is deployed on **Streamlit Community Cloud**:

**<https://mlzoomcampmidterm-ngv5vx8ao6wqc3gw5tpsyy.streamlit.app/>**

Deployment settings:

- Repository: `DemidenkoD/ml_zoomcamp_midterm`
- Branch: `master`
- Main file path: `app/app.py`
- Python dependencies: `requirements.txt` (runtime-only: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`)
- Model artifact: `models/final_pipeline.joblib` is committed, so no training is needed at startup

Note: Streamlit Community Cloud puts idle apps to sleep; the first visit may take a few seconds while the app wakes up.

---

## Notebook

`notebooks/notebook_main.ipynb` contains the full, executed workflow:

1. Setup and imports
2. Load data
3. Data quality check (missing values, duplicates, IDs)
4. Target distribution and correlation analysis
5. Train / validation / test split
6. Preprocessing and a scoring helper
7. Baseline models (Logistic Regression, Decision Tree)
8. Random Forest tuning with `GridSearchCV`
9. Final model, test score, and saving the artifact
10. Explainability with permutation importance and a confusion matrix
11. Summary

---

## Workflow Roadmap

1. [x] Project setup (environment, dependencies, dataset, folder structure)
2. [x] Exploratory Data Analysis (`notebooks/notebook_main.ipynb`, sections 2-4)
3. [x] Problem definition and validation strategy
4. [x] Baseline model (Logistic Regression)
5. [x] Model experiments and comparison (Logistic Regression, Decision Tree, Random Forest)
6. [x] Hyperparameter tuning (`GridSearchCV`)
7. [x] Model analysis and explainability (permutation importance, confusion matrix)
8. [x] Reproducible training and inference scripts (`src/train.py`, `src/predict.py`)
9. [x] Interactive Streamlit UI (`app/app.py`)
10. [x] Containerization with Docker
11. [x] Cloud deployment with Streamlit Community Cloud

---

## Limitations

- The features carry almost no signal about the self-reported stress score, so predictions are weak (macro F1 about 0.21).
- The target is self-reported, which is noisy by nature.
- `Wearable_Stress_Score` was excluded because it is an objective stress measurement and would leak the target.
- The tool is for exploration and education only, not for medical or psychological assessment.
