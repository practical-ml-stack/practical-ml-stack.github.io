# Practical ML Stack - Implementation Tasks

## Overview

This document tracks the implementation progress for building the Practical ML Stack web book using MkDocs Material.

---

## Phase 1: MkDocs Setup

| Task | File | Status |
|------|------|--------|
| 1.1 | Create `mkdocs.yml` with theme, plugins, and navigation config | ✅ Completed |
| 1.2 | Create `requirements.txt` with MkDocs Material dependencies | ✅ Completed |
| 1.3 | Create `docs/` folder structure | ✅ Completed |

---

## Phase 2: Homepage

| Task | File | Status |
|------|------|--------|
| 2.1 | Create `docs/index.md` with introduction and value proposition | ⬜ Pending |

---

## Phase 3: Getting Started

| Task | File | Status |
|------|------|--------|
| 3.1 | Create `docs/getting-started/index.md` (prerequisites, audience) | ⬜ Pending |
| 3.2 | Create `docs/getting-started/environment.md` (setup guide) | ⬜ Pending |

---

## Phase 4: Churn Modelling Use Case

| Task | File | Status |
|------|------|--------|
| 4.1 | Create `docs/use-cases/index.md` (overview of all use cases) | ⬜ Pending |
| 4.2 | Create `docs/use-cases/churn-modelling/index.md` (problem overview) | ⬜ Pending |
| 4.3 | Create `docs/use-cases/churn-modelling/data.md` (data understanding) | ⬜ Pending |
| 4.4 | Create `docs/use-cases/churn-modelling/features.md` (feature engineering) | ⬜ Pending |
| 4.5 | Create `docs/use-cases/churn-modelling/modelling.md` (model building) | ⬜ Pending |
| 4.6 | Create `docs/use-cases/churn-modelling/deployment.md` (production considerations) | ⬜ Pending |

---

## Phase 5: Jupyter Notebook

| Task | File | Status |
|------|------|--------|
| 5.1 | Create `notebooks/churn-modelling.ipynb` with Colab badge integration | ⬜ Pending |

---

## Phase 6: Contributors Section

| Task | File | Status |
|------|------|--------|
| 6.1 | Create `docs/contributors/index.md` (hub + how to contribute) | ⬜ Pending |
| 6.2 | Create `docs/contributors/profiles/template.md` (contributor template) | ⬜ Pending |

---

## Phase 7: Resources

| Task | File | Status |
|------|------|--------|
| 7.1 | Create `docs/resources/datasets.md` with curated dataset links | ⬜ Pending |
| 7.2 | Create `docs/resources/tools.md` with recommended tools | ⬜ Pending |

---

## Phase 8: GitHub Actions Deployment

| Task | File | Status |
|------|------|--------|
| 8.1 | Create `.github/workflows/deploy.yml` for auto-deployment to GitHub Pages | ⬜ Pending |

---

## Phase 9: Cleanup

| Task | File | Status |
|------|------|--------|
| 9.1 | Remove old `index.html` placeholder | ⬜ Pending |
| 9.2 | Update `README.md` with project info and setup instructions | ⬜ Pending |

---

## Status Legend

- ⬜ Pending
- 🔄 In Progress
- ✅ Completed
- ❌ Cancelled

---

## Final Site Structure

```
practical-ml-stack.github.io/
├── mkdocs.yml
├── requirements.txt
├── tasks.md
├── README.md
├── LICENSE
├── docs/
│   ├── index.md
│   ├── getting-started/
│   │   ├── index.md
│   │   └── environment.md
│   ├── use-cases/
│   │   ├── index.md
│   │   └── churn-modelling/
│   │       ├── index.md
│   │       ├── data.md
│   │       ├── features.md
│   │       ├── modelling.md
│   │       └── deployment.md
│   ├── contributors/
│   │   ├── index.md
│   │   └── profiles/
│   │       └── template.md
│   └── resources/
│       ├── datasets.md
│       └── tools.md
├── notebooks/
│   └── churn-modelling.ipynb
└── .github/
    └── workflows/
        └── deploy.yml
```

