---
title: Tools & Libraries
description: Recommended tools and libraries for building production ML systems. From experimentation to deployment.
---

# Tools & Libraries

A curated list of tools and libraries used in real ML workflows. Organized by stage of the ML lifecycle.

---

## Core Data Science Stack

### Essential Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| [pandas](https://pandas.pydata.org/) | Data manipulation | `pip install pandas` |
| [NumPy](https://numpy.org/) | Numerical computing | `pip install numpy` |
| [scikit-learn](https://scikit-learn.org/) | ML algorithms | `pip install scikit-learn` |
| [Matplotlib](https://matplotlib.org/) | Visualization | `pip install matplotlib` |
| [Seaborn](https://seaborn.pydata.org/) | Statistical visualization | `pip install seaborn` |

### Gradient Boosting

| Library | Strengths | Install |
|---------|-----------|---------|
| [XGBoost](https://xgboost.readthedocs.io/) | Speed, accuracy, widely used | `pip install xgboost` |
| [LightGBM](https://lightgbm.readthedocs.io/) | Very fast, handles large data | `pip install lightgbm` |
| [CatBoost](https://catboost.ai/) | Best for categorical features | `pip install catboost` |

!!! tip "Which to choose?"
    - **XGBoost**: Best default choice, most documentation
    - **LightGBM**: When speed matters or data is large
    - **CatBoost**: When you have many categorical features

---

## Experiment Tracking

Track experiments, compare runs, and reproduce results.

| Tool | Type | Best For |
|------|------|----------|
| [MLflow](https://mlflow.org/) | Open source | Local/team use, model registry |
| [Weights & Biases](https://wandb.ai/) | Cloud/self-hosted | Visualization, collaboration |
| [Neptune.ai](https://neptune.ai/) | Cloud | Team collaboration |
| [DVC](https://dvc.org/) | Open source | Data versioning + experiments |

### MLflow Quick Start

```python
import mlflow

# Start experiment
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "xgboost")
    mlflow.log_param("max_depth", 5)
    
    # Train model...
    
    # Log metrics
    mlflow.log_metric("auc", 0.85)
    mlflow.log_metric("precision", 0.72)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

---

## Feature Engineering

### Feature Stores

| Tool | Type | Best For |
|------|------|----------|
| [Feast](https://feast.dev/) | Open source | Getting started, flexible |
| [Tecton](https://www.tecton.ai/) | Managed | Enterprise, real-time |
| [Hopsworks](https://www.hopsworks.ai/) | Open source + managed | Full platform |

### Feature Libraries

| Library | Purpose | Install |
|---------|---------|---------|
| [Feature-engine](https://feature-engine.readthedocs.io/) | Sklearn-compatible transformers | `pip install feature-engine` |
| [category_encoders](https://contrib.scikit-learn.org/category_encoders/) | Categorical encoding | `pip install category_encoders` |
| [tsfresh](https://tsfresh.readthedocs.io/) | Time series features | `pip install tsfresh` |

---

## Model Deployment

### Serving Frameworks

| Tool | Type | Best For |
|------|------|----------|
| [FastAPI](https://fastapi.tiangolo.com/) | Web framework | Simple REST APIs |
| [BentoML](https://www.bentoml.com/) | ML serving | Standardized serving |
| [Seldon](https://www.seldon.io/) | Kubernetes-native | Scale, enterprise |
| [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) | TF models | High performance |

### FastAPI Example

```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: dict):
    prediction = model.predict([list(features.values())])
    return {"prediction": prediction[0]}
```

---

## Data Validation

| Tool | Purpose | Install |
|------|---------|---------|
| [Great Expectations](https://greatexpectations.io/) | Data quality testing | `pip install great_expectations` |
| [Pandera](https://pandera.readthedocs.io/) | DataFrame validation | `pip install pandera` |
| [pydantic](https://pydantic-docs.helpmanual.io/) | Data validation | `pip install pydantic` |

### Pandera Example

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "tenure": pa.Column(int, pa.Check.ge(0)),
    "monthly_charges": pa.Column(float, pa.Check.between(0, 500)),
    "churn": pa.Column(int, pa.Check.isin([0, 1]))
})

# Validate DataFrame
schema.validate(df)
```

---

## Workflow Orchestration

| Tool | Type | Best For |
|------|------|----------|
| [Apache Airflow](https://airflow.apache.org/) | Open source | General workflows |
| [Prefect](https://www.prefect.io/) | Open source + cloud | Modern, Pythonic |
| [Dagster](https://dagster.io/) | Open source | Data-aware orchestration |
| [Kubeflow Pipelines](https://www.kubeflow.org/) | Kubernetes | ML-specific pipelines |

---

## Model Monitoring

| Tool | Type | Best For |
|------|------|----------|
| [Evidently](https://www.evidentlyai.com/) | Open source | Data/model drift |
| [Whylogs](https://whylabs.ai/whylogs) | Open source | Data profiling |
| [NannyML](https://www.nannyml.com/) | Open source | Performance monitoring |
| [Arize](https://arize.com/) | Managed | Full observability |

### Evidently Example

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=production_df)
report.save_html("drift_report.html")
```

---

## Development Environment

### IDEs

| Tool | Best For |
|------|----------|
| [VS Code](https://code.visualstudio.com/) | General development, notebooks |
| [PyCharm](https://www.jetbrains.com/pycharm/) | Pure Python projects |
| [JupyterLab](https://jupyter.org/) | Exploration, notebooks |

### Environment Management

| Tool | Purpose | Install |
|------|---------|---------|
| [conda](https://docs.conda.io/) | Environment + package management | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) |
| [venv](https://docs.python.org/3/library/venv.html) | Built-in virtual environments | Included in Python |
| [uv](https://github.com/astral-sh/uv) | Fast pip replacement | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Poetry](https://python-poetry.org/) | Dependency management | `pip install poetry` |

---

## Recommended Stack by Stage

### :material-school: Learning

```
pandas + scikit-learn + matplotlib + Jupyter
```

### :material-account-group: Team Projects

```
+ MLflow + DVC + Great Expectations
```

### :material-factory: Production

```
+ Airflow/Prefect + FastAPI/BentoML + Evidently
```

---

## Learning Resources

### Books

- *Hands-On Machine Learning* by Aurélien Géron
- *Designing Machine Learning Systems* by Chip Huyen
- *Machine Learning Engineering* by Andriy Burkov

### Courses

- [Fast.ai](https://www.fast.ai/) - Practical deep learning
- [Made With ML](https://madewithml.com/) - MLOps
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) - Production ML

### Blogs

- [Chip Huyen's Blog](https://huyenchip.com/blog/)
- [Eugene Yan's Blog](https://eugeneyan.com/)
- [ML Ops Community](https://mlops.community/)
