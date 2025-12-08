---
title: Data Understanding - Churn Modelling
description: Exploratory data analysis for customer churn prediction. Understand your data before building models.
---

# Data Understanding

<div class="badge-container" markdown>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/practical-ml-stack/practical-ml-stack.github.io/blob/main/notebooks/churn-modelling.ipynb)

</div>

Before building any model, you need to deeply understand your data. This section covers loading the data, exploratory analysis, and identifying data quality issues.

---

## Loading the Data

First, let's load the Telco Customer Churn dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

**Output:**
```
Dataset shape: (7043, 21)
Columns: ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
          'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
          'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
          'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
```

---

## Quick Data Overview

```python
# Basic info
df.info()
```

```
<class 'pandas.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 ...
 20  Churn             7043 non-null   object 
```

```python
# First few rows
df.head()
```

| customerID | gender | SeniorCitizen | Partner | tenure | MonthlyCharges | Churn |
|------------|--------|---------------|---------|--------|----------------|-------|
| 7590-VHVEG | Female | 0 | Yes | 1 | 29.85 | No |
| 5575-GNVDE | Male | 0 | No | 34 | 56.95 | No |
| 3668-QPYBK | Male | 0 | No | 2 | 53.85 | Yes |

---

## Understanding the Target Variable

```python
# Churn distribution
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print("Churn Distribution:")
print(f"  No:  {churn_counts['No']:,} ({churn_pct['No']:.1f}%)")
print(f"  Yes: {churn_counts['Yes']:,} ({churn_pct['Yes']:.1f}%)")
```

**Output:**
```
Churn Distribution:
  No:  5,174 (73.5%)
  Yes: 1,869 (26.5%)
```

```python
# Visualize
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
df['Churn'].value_counts().plot(kind='bar', color=colors, ax=ax)
ax.set_title('Churn Distribution', fontsize=14)
ax.set_xlabel('Churn')
ax.set_ylabel('Count')
ax.set_xticklabels(['No', 'Yes'], rotation=0)

# Add percentages
for i, (count, pct) in enumerate(zip(churn_counts, churn_pct)):
    ax.text(i, count + 100, f'{pct:.1f}%', ha='center', fontsize=12)

plt.tight_layout()
plt.show()
```

!!! info "Class Imbalance"
    With ~27% churn rate, we have moderate class imbalance. This is actually realistic for many businesses. We'll address this during modeling.

---

## Feature Categories

Let's organize features by type:

### Demographic Features

```python
demographic_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

for col in demographic_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())
```

| Feature | Values | Notes |
|---------|--------|-------|
| gender | Male (50.5%), Female (49.5%) | Balanced |
| SeniorCitizen | No (83.8%), Yes (16.2%) | Binary (0/1) |
| Partner | No (51.7%), Yes (48.3%) | Balanced |
| Dependents | No (70.0%), Yes (30.0%) | Slight imbalance |

### Service Features

```python
service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']

# Count services per customer
df['num_services'] = df[service_cols].apply(
    lambda x: sum(x.isin(['Yes', 'Fiber optic', 'DSL'])), axis=1
)
```

### Account Features

```python
account_cols = ['Contract', 'PaperlessBilling', 'PaymentMethod', 
                'MonthlyCharges', 'TotalCharges', 'tenure']

# Contract distribution
print(df['Contract'].value_counts())
```

```
Month-to-month    3875
Two year          1695
One year          1473
```

!!! warning "Key Insight"
    Month-to-month contracts are the most common—and likely the highest churn risk. We'll verify this shortly.

---

## Data Quality Issues

### Missing Values

```python
# Check for missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

print("Missing Values:")
print(missing[missing > 0])
```

```
Missing Values:
Series([], dtype: int64)
```

Looks clean! But wait...

```python
# Check for hidden missing values
print(df['TotalCharges'].dtype)  # object - should be numeric!

# Find non-numeric values
non_numeric = df[pd.to_numeric(df['TotalCharges'], errors='coerce').isna()]
print(f"Non-numeric TotalCharges: {len(non_numeric)} rows")
print(non_numeric[['customerID', 'tenure', 'TotalCharges']].head())
```

```
Non-numeric TotalCharges: 11 rows
     customerID  tenure TotalCharges
488  4472-LVYGI       0             
753  3115-CZMZD       0             
...
```

!!! bug "Data Quality Issue Found"
    11 customers have blank `TotalCharges`. These are all new customers (tenure=0). The blank likely means $0.

**Fix:**

```python
# Convert TotalCharges to numeric, fill blanks with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
```

### Duplicate Check

```python
# Check for duplicate customer IDs
duplicates = df['customerID'].duplicated().sum()
print(f"Duplicate customer IDs: {duplicates}")
```

```
Duplicate customer IDs: 0
```

---

## Exploratory Analysis

### Churn by Contract Type

```python
# Churn rate by contract
churn_by_contract = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).sort_values(ascending=False)

print("Churn Rate by Contract:")
print(churn_by_contract.round(1))
```

```
Churn Rate by Contract:
Month-to-month    42.7%
One year          11.3%
Two year           2.8%
```

!!! success "Key Finding"
    Month-to-month customers churn at **15x the rate** of two-year contract customers. Contract type will be a powerful predictor.

### Churn by Tenure

```python
# Tenure distribution by churn
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
for churn_val, color in [('No', '#2ecc71'), ('Yes', '#e74c3c')]:
    subset = df[df['Churn'] == churn_val]
    axes[0].hist(subset['tenure'], bins=30, alpha=0.6, 
                 label=f'Churn={churn_val}', color=color)
axes[0].set_xlabel('Tenure (months)')
axes[0].set_ylabel('Count')
axes[0].set_title('Tenure Distribution by Churn')
axes[0].legend()

# Churn rate by tenure bucket
df['tenure_bucket'] = pd.cut(df['tenure'], 
                             bins=[0, 12, 24, 48, 72], 
                             labels=['0-12', '13-24', '25-48', '49-72'])
churn_by_tenure = df.groupby('tenure_bucket')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
)
axes[1].bar(churn_by_tenure.index, churn_by_tenure.values, color='#3498db')
axes[1].set_xlabel('Tenure (months)')
axes[1].set_ylabel('Churn Rate (%)')
axes[1].set_title('Churn Rate by Tenure')

plt.tight_layout()
plt.show()
```

!!! success "Key Finding"
    New customers (0-12 months) have the highest churn rate (~47%). Churn risk decreases significantly with tenure.

### Churn by Monthly Charges

```python
# Monthly charges distribution
fig, ax = plt.subplots(figsize=(10, 5))

df[df['Churn'] == 'No']['MonthlyCharges'].hist(
    bins=30, alpha=0.6, label='No Churn', color='#2ecc71', ax=ax
)
df[df['Churn'] == 'Yes']['MonthlyCharges'].hist(
    bins=30, alpha=0.6, label='Churned', color='#e74c3c', ax=ax
)
ax.set_xlabel('Monthly Charges ($)')
ax.set_ylabel('Count')
ax.set_title('Monthly Charges Distribution by Churn')
ax.legend()
plt.show()

# Average charges
print(f"Avg Monthly Charges - Churned: ${df[df['Churn']=='Yes']['MonthlyCharges'].mean():.2f}")
print(f"Avg Monthly Charges - Retained: ${df[df['Churn']=='No']['MonthlyCharges'].mean():.2f}")
```

```
Avg Monthly Charges - Churned: $74.44
Avg Monthly Charges - Retained: $61.27
```

!!! success "Key Finding"
    Churned customers pay **$13 more per month** on average. Higher-paying customers may have higher expectations or more alternatives.

---

## Correlation Analysis

```python
# Encode categorical variables for correlation
df_encoded = df.copy()
df_encoded['Churn'] = (df_encoded['Churn'] == 'Yes').astype(int)

# Select numeric columns
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
correlation = df_encoded[numeric_cols].corr()['Churn'].sort_values(ascending=False)

print("Correlation with Churn:")
print(correlation.round(3))
```

```
Correlation with Churn:
Churn             1.000
MonthlyCharges    0.193
SeniorCitizen     0.150
TotalCharges     -0.198
tenure           -0.352
```

!!! info "Interpretation"
    - **Positive correlation**: Higher monthly charges and being a senior citizen are associated with higher churn
    - **Negative correlation**: Longer tenure and higher total charges (loyal customers) are associated with lower churn

---

## Key Takeaways

From our EDA, we've identified the most important factors:

| Factor | Impact on Churn | Strength |
|--------|-----------------|----------|
| Contract Type | Month-to-month = high risk | :material-star::material-star::material-star: Very Strong |
| Tenure | New customers = high risk | :material-star::material-star::material-star: Very Strong |
| Monthly Charges | Higher charges = higher risk | :material-star::material-star: Moderate |
| Internet Service | Fiber optic = higher risk | :material-star::material-star: Moderate |
| Senior Citizen | Seniors = slightly higher risk | :material-star: Weak |

---

## Data Preparation Summary

```python
# Final data preparation
def prepare_data(df):
    """Prepare raw data for modeling."""
    df = df.copy()
    
    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert target to binary
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Drop customerID (not predictive)
    df = df.drop('customerID', axis=1)
    
    return df

df_clean = prepare_data(df)
print(f"Clean data shape: {df_clean.shape}")
```

---

## Next Steps

Now that we understand our data, let's engineer features that capture customer behavior patterns.

<div class="use-case-grid" markdown>

<div class="use-case-card" markdown>

### :material-arrow-left: Previous

Review the problem overview.

[:octicons-arrow-left-24: Problem Overview](index.md)

</div>

<div class="use-case-card" markdown>

### :material-arrow-right: Next

Create predictive features from raw data.

[:octicons-arrow-right-24: Feature Engineering](features.md){ .md-button .md-button--primary }

</div>

</div>
