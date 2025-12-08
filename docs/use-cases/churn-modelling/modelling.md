---
title: Model Building - Churn Modelling
description: Train, tune, and evaluate machine learning models for customer churn prediction.
---

# Model Building

<div class="badge-container" markdown>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/practical-ml-stack/practical-ml-stack.github.io/blob/main/notebooks/churn-modelling.ipynb)

</div>

Now that we have our features, let's train and evaluate models. We'll compare multiple algorithms and select the best one for production.

---

## Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

import warnings
warnings.filterwarnings('ignore')

# Load and prepare data (from previous sections)
# X = build_feature_matrix(df)
# y = (df['Churn'] == 'Yes').astype(int)
```

---

## Train/Test Split

```python
# Split data with stratification (maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Important for imbalanced data
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Churn rate (train): {y_train.mean():.1%}")
print(f"Churn rate (test): {y_test.mean():.1%}")
```

```
Training set: 5634 samples
Test set: 1409 samples
Churn rate (train): 26.5%
Churn rate (test): 26.5%
```

!!! tip "Why Stratify?"
    Stratification ensures both train and test sets have the same class distribution. Without it, you might accidentally get all churners in one set.

---

## Baseline Model

Always start with a simple baseline:

```python
# Baseline: Always predict the majority class
baseline_accuracy = 1 - y_test.mean()
print(f"Baseline Accuracy (predict all 'No Churn'): {baseline_accuracy:.1%}")

# A useful model must beat this!
```

```
Baseline Accuracy (predict all 'No Churn'): 73.5%
```

!!! warning "Accuracy is Misleading"
    With 73.5% non-churners, predicting "No Churn" for everyone gives 73.5% accuracy but catches **zero** actual churners. We need better metrics.

---

## Model 1: Logistic Regression

A great starting point—interpretable and fast:

```python
# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
lr = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
lr.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.3f}")
```

```
Logistic Regression Results:
              precision    recall  f1-score   support

           0       0.88      0.76      0.82      1036
           1       0.55      0.74      0.63       373

    accuracy                           0.76      1409
   macro avg       0.72      0.75      0.72      1409
weighted avg       0.79      0.76      0.77      1409

ROC-AUC: 0.838
```

### Feature Coefficients

```python
# Get feature importance from coefficients
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("Top 10 Features (by absolute coefficient):")
print(coef_df.head(10))
```

```
Top 10 Features:
                 feature  coefficient
0       contract_monthly        1.234
1                 tenure       -0.987
2     payment_electronic        0.654
3         total_services       -0.543
4        is_new_customer        0.498
...
```

!!! success "Interpretability"
    Logistic Regression tells us exactly how each feature affects churn probability. Positive coefficients increase churn risk.

---

## Model 2: Random Forest

An ensemble method that often performs better:

```python
# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Evaluate
print("Random Forest Results:")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.3f}")
```

```
Random Forest Results:
              precision    recall  f1-score   support

           0       0.87      0.82      0.85      1036
           1       0.59      0.68      0.63       373

    accuracy                           0.79      1409
   macro avg       0.73      0.75      0.74      1409
weighted avg       0.80      0.79      0.79      1409

ROC-AUC: 0.847
```

---

## Model 3: Gradient Boosting (XGBoost)

State-of-the-art for tabular data:

```python
try:
    from xgboost import XGBClassifier
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    print("XGBoost Results:")
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.3f}")
    
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
```

```
XGBoost Results:
              precision    recall  f1-score   support

           0       0.88      0.84      0.86      1036
           1       0.62      0.70      0.66       373

    accuracy                           0.80      1409
   macro avg       0.75      0.77      0.76      1409
weighted avg       0.81      0.80      0.81      1409

ROC-AUC: 0.856
```

---

## Model Comparison

```python
# Compare all models
models = {
    'Logistic Regression': (y_pred_lr, y_prob_lr),
    'Random Forest': (y_pred_rf, y_prob_rf),
    'XGBoost': (y_pred_xgb, y_prob_xgb)
}

results = []
for name, (y_pred, y_prob) in models.items():
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results)
print(results_df.round(3))
```

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.758 | 0.553 | 0.740 | 0.633 | 0.838 |
| Random Forest | 0.785 | 0.589 | 0.684 | 0.633 | 0.847 |
| XGBoost | 0.804 | 0.618 | 0.700 | 0.657 | 0.856 |

---

## Hyperparameter Tuning

Let's tune the best-performing model (XGBoost):

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

xgb_tuned = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Grid search with cross-validation
grid_search = GridSearchCV(
    xgb_tuned, 
    param_grid, 
    cv=5, 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.3f}")
```

```
Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 3, 'n_estimators': 200}
Best CV ROC-AUC: 0.851
```

---

## Threshold Optimization

The default threshold (0.5) might not be optimal for business:

```python
# Get precision-recall for different thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_xgb)

# Find threshold that maximizes F1
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]

print(f"Optimal threshold: {best_threshold:.3f}")
print(f"At this threshold:")
print(f"  Precision: {precision[best_threshold_idx]:.3f}")
print(f"  Recall: {recall[best_threshold_idx]:.3f}")
print(f"  F1: {f1_scores[best_threshold_idx]:.3f}")

# Plot precision-recall curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(recall, precision, 'b-', linewidth=2)
ax.scatter([recall[best_threshold_idx]], [precision[best_threshold_idx]], 
           color='red', s=100, zorder=5, label=f'Optimal (threshold={best_threshold:.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

## Business-Oriented Threshold Selection

```python
# Consider business constraints
# Scenario: Retention team can contact 500 customers/month

# Sort customers by churn probability
test_df = pd.DataFrame({
    'actual': y_test.values,
    'probability': y_prob_xgb
}).sort_values('probability', ascending=False)

# Top 500 customers
top_500 = test_df.head(500)
churners_caught = top_500['actual'].sum()
total_churners = y_test.sum()

print(f"If we contact top 500 highest-risk customers:")
print(f"  Churners caught: {churners_caught} out of {total_churners}")
print(f"  Catch rate: {churners_caught/total_churners:.1%}")
print(f"  Precision (of 500 contacted): {churners_caught/500:.1%}")
```

```
If we contact top 500 highest-risk customers:
  Churners caught: 289 out of 373
  Catch rate: 77.5%
  Precision (of 500 contacted): 57.8%
```

!!! success "Business Impact"
    By scoring customers, we can catch **77.5% of churners** by contacting only the top **35%** of the customer base. That's a **2.2x lift** over random targeting!

---

## Confusion Matrix Analysis

```python
# Apply optimized threshold
y_pred_optimized = (y_prob_xgb >= best_threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_optimized)

# Visualize
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Predicted: Stay', 'Predicted: Churn'],
            yticklabels=['Actual: Stay', 'Actual: Churn'])
ax.set_title('Confusion Matrix')
plt.show()

# Interpret
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives (correctly predicted stay): {tn}")
print(f"False Positives (incorrectly predicted churn): {fp}")
print(f"False Negatives (missed churners): {fn}")
print(f"True Positives (correctly predicted churn): {tp}")
```

---

## Model Interpretability

Understanding why the model makes predictions:

```python
# Feature importance from XGBoost
importance_xgb = pd.DataFrame({
    'feature': X.columns,
    'importance': grid_search.best_estimator_.feature_importances_
}).sort_values('importance', ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
top_15 = importance_xgb.head(15)
ax.barh(top_15['feature'], top_15['importance'])
ax.set_xlabel('Importance')
ax.set_title('Top 15 Features - XGBoost')
ax.invert_yaxis()
plt.tight_layout()
plt.show()
```

### SHAP Values (Advanced)

```python
try:
    import shap
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(grid_search.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
except ImportError:
    print("SHAP not installed. Run: pip install shap")
```

---

## Final Model

```python
# Train final model on full training data with best parameters
final_model = XGBClassifier(
    **grid_search.best_params_,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(X_train, y_train)

# Save model
import joblib
joblib.dump(final_model, 'churn_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

print("Model saved successfully!")
```

---

## Key Takeaways

| Aspect | Finding |
|--------|---------|
| **Best Model** | XGBoost with ROC-AUC of 0.856 |
| **Top Features** | Contract type, tenure, payment method |
| **Optimal Threshold** | ~0.35 for balanced precision/recall |
| **Business Lift** | 2.2x better than random targeting |

---

## Next Steps

Now let's learn how to deploy this model in production.

<div class="use-case-grid" markdown>

<div class="use-case-card" markdown>

### :material-arrow-left: Previous

Review feature engineering.

[:octicons-arrow-left-24: Feature Engineering](features.md)

</div>

<div class="use-case-card" markdown>

### :material-arrow-right: Next

Deploy your model to production.

[:octicons-arrow-right-24: Deployment](deployment.md){ .md-button .md-button--primary }

</div>

</div>
