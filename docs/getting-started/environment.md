---
title: Environment Setup
description: Set up your local development environment for running Practical ML Stack notebooks and experiments.
---

# Environment Setup

This guide walks you through setting up a local development environment for running the notebooks and experimenting with your own data.

!!! tip "Just Want to Try It?"
    You can skip local setup entirely and use **Google Colab**. Every use case has an "Open in Colab" button that runs in your browser with no installation required.

---

## Quick Setup (5 minutes)

For those who know what they're doing:

=== "pip"

    ```bash
    # Create virtual environment
    python -m venv practical-ml-env
    source practical-ml-env/bin/activate  # On Windows: practical-ml-env\Scripts\activate

    # Install dependencies
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    pip install xgboost lightgbm catboost  # Optional: gradient boosting
    
    # Launch Jupyter
    jupyter notebook
    ```

=== "conda"

    ```bash
    # Create conda environment
    conda create -n practical-ml python=3.10
    conda activate practical-ml

    # Install dependencies
    conda install pandas numpy scikit-learn matplotlib seaborn jupyter
    conda install -c conda-forge xgboost lightgbm catboost  # Optional
    
    # Launch Jupyter
    jupyter notebook
    ```

=== "uv (fast)"

    ```bash
    # Install uv if you haven't
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create environment and install
    uv venv practical-ml-env
    source practical-ml-env/bin/activate
    uv pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    
    # Launch Jupyter
    jupyter notebook
    ```

---

## Detailed Setup Guide

### Step 1: Install Python

We recommend **Python 3.10 or 3.11** for best compatibility.

=== "macOS"

    ```bash
    # Using Homebrew (recommended)
    brew install python@3.11
    
    # Verify installation
    python3 --version
    ```

    Or download from [python.org](https://www.python.org/downloads/).

=== "Windows"

    1. Download Python from [python.org](https://www.python.org/downloads/)
    2. Run the installer
    3. **Important**: Check "Add Python to PATH"
    4. Verify in Command Prompt:
    
    ```cmd
    python --version
    ```

=== "Linux"

    ```bash
    # Ubuntu/Debian
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3-pip
    
    # Verify installation
    python3 --version
    ```

### Step 2: Create a Virtual Environment

!!! warning "Why Virtual Environments?"
    Virtual environments keep your project dependencies isolated. Different projects can use different package versions without conflicts.

```bash
# Navigate to your projects folder
cd ~/projects  # or wherever you keep your code

# Create a new directory for this work
mkdir practical-ml-stack
cd practical-ml-stack

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install Core Dependencies

Create a `requirements.txt` file with the following contents:

```txt title="requirements.txt"
# Core data science
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Jupyter
jupyter>=1.0.0
jupyterlab>=4.0.0
notebook>=7.0.0

# Gradient Boosting (optional but recommended)
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# Utilities
tqdm>=4.65.0
python-dotenv>=1.0.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

Run this quick test to make sure everything works:

```python
# test_setup.py
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"scikit-learn: {sklearn.__version__}")

# Quick test
df = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
print(f"\nTest DataFrame:\n{df}")
print("\n✅ All good! Your environment is ready.")
```

```bash
python test_setup.py
```

### Step 5: Launch Jupyter

```bash
# Classic Notebook interface
jupyter notebook

# OR JupyterLab (more features)
jupyter lab
```

Your browser should open automatically. If not, look for a URL in the terminal output like:
```
http://localhost:8888/?token=abc123...
```

---

## IDE Setup (Optional)

### VS Code

VS Code with the Python extension is excellent for ML work:

1. Install [VS Code](https://code.visualstudio.com/)
2. Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
3. Install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
4. Open your project folder
5. Select your virtual environment (bottom-left status bar)

### PyCharm

1. Install [PyCharm](https://www.jetbrains.com/pycharm/) (Community edition is free)
2. Open your project folder
3. Configure interpreter: Settings → Project → Python Interpreter
4. Select your virtual environment

---

## Google Colab Alternative

If you prefer not to set up a local environment:

### Pros of Colab

- :material-check: No installation required
- :material-check: Free GPU/TPU access
- :material-check: Pre-installed ML libraries
- :material-check: Easy sharing and collaboration

### Cons of Colab

- :material-close: Requires internet connection
- :material-close: Sessions timeout after inactivity
- :material-close: Limited customization
- :material-close: Slower for small operations (network overhead)

### Using Colab with Our Notebooks

1. Click the "Open in Colab" badge on any use case page
2. The notebook will open in Colab
3. Click "Copy to Drive" to save your own version
4. Run cells with Shift+Enter

---

## Troubleshooting

### Common Issues

??? question "pip install fails with permission error"
    
    Make sure you've activated your virtual environment:
    ```bash
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```
    
    If still failing, try:
    ```bash
    pip install --user package-name
    ```

??? question "Jupyter can't find my virtual environment"
    
    Install the IPython kernel in your environment:
    ```bash
    pip install ipykernel
    python -m ipykernel install --user --name=practical-ml
    ```
    
    Then select "practical-ml" as the kernel in Jupyter.

??? question "Import errors after installation"
    
    Make sure you're running Jupyter from within your virtual environment:
    ```bash
    # Activate environment first
    source venv/bin/activate
    
    # Then launch Jupyter
    jupyter notebook
    ```

??? question "matplotlib plots not showing"
    
    In Jupyter, add this at the top of your notebook:
    ```python
    %matplotlib inline
    ```

---

## Package Versions Reference

These are the versions we've tested with:

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10, 3.11 | 3.12 may have compatibility issues |
| pandas | 2.0+ | Major API changes from 1.x |
| scikit-learn | 1.3+ | |
| XGBoost | 2.0+ | New API in 2.0 |
| LightGBM | 4.0+ | |

---

## Next Steps

Your environment is ready! Time to start learning:

<div class="use-case-grid" markdown>

<div class="use-case-card" markdown>

### :material-account-cancel: Churn Modelling

Start with our most complete use case.

[:octicons-arrow-right-24: Start Learning](../use-cases/churn-modelling/index.md)

</div>

<div class="use-case-card" markdown>

### :material-database: Explore Datasets

Find real-world datasets to practice with.

[:octicons-arrow-right-24: Dataset Resources](../resources/datasets.md)

</div>

</div>
