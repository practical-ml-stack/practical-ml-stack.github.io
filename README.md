# Practical ML Stack

[![Deploy](https://github.com/practical-ml-stack/practical-ml-stack.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/practical-ml-stack/practical-ml-stack.github.io/actions/workflows/deploy.yml)

**Real-world machine learning solutions for industry problems.**

Move beyond toy datasets. Build production-ready ML systems that solve real business problems.

🌐 **Website**: [https://practical-ml-stack.github.io/](https://practical-ml-stack.github.io/)

---

## What is Practical ML Stack?

A free, open-source resource for ML practitioners who want to bridge the gap between tutorial ML and production ML. 

**What you'll learn:**

- 🎯 **Real use cases**: Churn prediction, demand forecasting, cross-sell modeling
- 🔧 **Practical techniques**: Feature engineering, model deployment, monitoring
- 💼 **Business context**: Why decisions matter, not just how to code them
- 🚀 **Production patterns**: From notebook to deployed system

---

## Use Cases

| Use Case | Status | Description |
|----------|--------|-------------|
| [Churn Modelling](https://practical-ml-stack.github.io/use-cases/churn-modelling/) | ✅ Complete | Predict customer churn with classification |
| Demand Forecasting | 🔜 Coming Soon | Time series forecasting for inventory |
| Cross-Sell Modelling | 🔜 Coming Soon | Recommendation and propensity models |
| Assortment Optimization | 🔜 Coming Soon | ML + optimization for retail |

---

## Local Development

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/practical-ml-stack/practical-ml-stack.github.io.git
cd practical-ml-stack.github.io

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start local server
mkdocs serve
```

Visit [http://localhost:8000](http://localhost:8000) to view the site.

### Build for Production

```bash
mkdocs build
```

The built site will be in the `site/` directory.

---

## Contributing

We welcome contributions from ML practitioners! See our [Contributing Guide](https://practical-ml-stack.github.io/contributors/) for details.

**Ways to contribute:**

- 📝 Add a new use case
- 🐛 Fix bugs or improve explanations
- 🌍 Translate content
- 💡 Suggest improvements

---

## Tech Stack

- **Framework**: [MkDocs](https://www.mkdocs.org/) with [Material theme](https://squidfunk.github.io/mkdocs-material/)
- **Hosting**: GitHub Pages
- **CI/CD**: GitHub Actions

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Inspired by [mlsysbook.ai](https://mlsysbook.ai/) and the broader ML education community.

Built with ❤️ by practitioners, for practitioners.
