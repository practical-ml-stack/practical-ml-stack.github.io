---
title: Datasets
description: Curated collection of real-world datasets for practicing machine learning. Find data for churn, forecasting, fraud detection, and more.
---

# Datasets

Finding the right dataset is often the first challenge in any ML project. This page curates high-quality, real-world datasets organized by use case.

---

## Dataset Sources

### :material-star: Primary Sources

<div class="dataset-grid" markdown>

<div class="use-case-card" markdown>

#### [Kaggle Datasets](https://www.kaggle.com/datasets)

The largest repository of ML datasets. Great for:

- Competition datasets with benchmarks
- Community-uploaded real-world data
- Notebooks showing how others approached problems

</div>

<div class="use-case-card" markdown>

#### [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)

Classic ML datasets used in academic research:

- Well-documented and clean
- Widely cited in literature
- Good for benchmarking

</div>

<div class="use-case-card" markdown>

#### [Google Dataset Search](https://datasetsearch.research.google.com/)

Search engine for datasets across the web:

- Aggregates from multiple sources
- Includes government and research data
- Good for finding niche datasets

</div>

<div class="use-case-card" markdown>

#### [OpenML](https://www.openml.org/)

Platform for sharing ML experiments:

- Standardized dataset format
- Includes benchmark results
- API for programmatic access

</div>

</div>

### :material-database: Other Sources

| Source | Best For | Notes |
|--------|----------|-------|
| [Hugging Face Datasets](https://huggingface.co/datasets) | NLP, Computer Vision | Growing collection, easy to load |
| [AWS Open Data](https://registry.opendata.aws/) | Large-scale data | Free to access, cloud-native |
| [Data.gov](https://data.gov/) | Government data | US federal data, various domains |
| [Zenodo](https://zenodo.org/) | Research data | Academic datasets with DOIs |
| [Figshare](https://figshare.com/) | Research data | Includes supplementary materials |
| [DataHub.io](https://datahub.io/) | Curated collections | Clean, well-documented |

---

## Datasets by Use Case

### :material-account-cancel: Customer Churn

| Dataset | Size | Source | Description |
|---------|------|--------|-------------|
| [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | 7K | Kaggle | Telecom churn with demographics and services |
| [Bank Customer Churn](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) | 10K | Kaggle | Banking churn with geography and products |
| [KKBox Churn](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) | 2.6M | Kaggle | Music streaming service |
| [E-Commerce Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) | 5K | Kaggle | Online retail churn |

### :material-chart-timeline-variant: Demand Forecasting

| Dataset | Size | Source | Description |
|---------|------|--------|-------------|
| [Store Sales - Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting) | 3M | Kaggle | Ecuadorian grocery store sales |
| [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) | 46M | Kaggle | Walmart sales, hierarchical |
| [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) | 1M | Kaggle | German drugstore chain |
| [Web Traffic Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) | 145K | Kaggle | Wikipedia page views |

### :material-credit-card: Fraud Detection

| Dataset | Size | Source | Description |
|---------|------|--------|-------------|
| [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 284K | Kaggle | Anonymized transactions, highly imbalanced |
| [IEEE-CIS Fraud](https://www.kaggle.com/c/ieee-fraud-detection) | 590K | Kaggle | E-commerce transactions |
| [Synthetic Fraud](https://www.kaggle.com/datasets/ealaxi/paysim1) | 6.3M | Kaggle | Simulated mobile money transactions |

### :material-cart-plus: Recommendations / Cross-Sell

| Dataset | Size | Source | Description |
|---------|------|--------|-------------|
| [Amazon Product Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) | 568K | Kaggle | Food reviews with ratings |
| [MovieLens](https://grouplens.org/datasets/movielens/) | 25M | GroupLens | Movie ratings, multiple sizes |
| [Instacart Orders](https://www.kaggle.com/c/instacart-market-basket-analysis) | 3.4M | Kaggle | Grocery orders with products |
| [Retail Transactions](https://www.kaggle.com/datasets/vijayuv/onlineretail) | 541K | Kaggle | UK online retail transactions |

### :material-currency-usd: Pricing / Revenue

| Dataset | Size | Source | Description |
|---------|------|--------|-------------|
| [Mercari Price Suggestion](https://www.kaggle.com/c/mercari-price-suggestion-challenge) | 1.4M | Kaggle | Product pricing |
| [Airbnb Listings](http://insideairbnb.com/get-the-data/) | Varies | Inside Airbnb | Rental prices by city |
| [Used Cars](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) | 426K | Kaggle | Vehicle pricing |

### :material-head-question: Customer Analytics

| Dataset | Size | Source | Description |
|---------|------|--------|-------------|
| [Marketing Campaign](https://www.kaggle.com/datasets/rodsaldanha/arketing-campaign) | 2K | Kaggle | Customer response to campaigns |
| [Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) | 200 | Kaggle | RFM-style data |
| [Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | 100K | Kaggle | Orders, reviews, sellers |

---

## How to Choose a Dataset

### For Learning

1. **Start small** (< 100K rows) - faster iteration
2. **Well-documented** - clear feature descriptions
3. **Known benchmarks** - compare your results
4. **Clean enough** - some data quality issues, but not overwhelming

### For Portfolio Projects

1. **Interesting domain** - something you can talk about in interviews
2. **Real-world messiness** - shows you can handle imperfect data
3. **Business relevance** - demonstrates you understand business problems
4. **Unique angle** - don't just reproduce existing notebooks

### For Production Practice

1. **Large scale** (> 1M rows) - practice with real volumes
2. **Multiple tables** - practice joins and feature engineering
3. **Time component** - practice proper train/test splits
4. **Ongoing updates** - practice with data pipelines

---

## Loading Datasets

### Kaggle API

```python
# Install and configure
# pip install kaggle
# Place kaggle.json in ~/.kaggle/

import kaggle

# Download dataset
kaggle.api.dataset_download_files(
    'blastchar/telco-customer-churn',
    path='./data',
    unzip=True
)
```

### OpenML

```python
from sklearn.datasets import fetch_openml

# Load dataset by name or ID
data = fetch_openml(name='credit-g', version=1, as_frame=True)
df = data.frame
```

### Hugging Face

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")
```

### Direct Download

```python
import pandas as pd

# Many Kaggle datasets have direct URLs
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
```

---

## Data Licensing

!!! warning "Check the License"
    Always check dataset licenses before using data, especially for:
    
    - Commercial projects
    - Publishing results
    - Redistributing data

Common licenses:

| License | Commercial Use | Attribution Required |
|---------|----------------|---------------------|
| CC0 | ✅ Yes | ❌ No |
| CC-BY | ✅ Yes | ✅ Yes |
| CC-BY-NC | ❌ No | ✅ Yes |
| Research Only | ❌ No | ✅ Yes |

---

## Request a Dataset

Can't find a dataset for your use case? [Open an issue](https://github.com/practical-ml-stack/practical-ml-stack.github.io/issues) and we'll try to help!
