# Eye Disease Classification — Ophthalmological Diagnostic Support System

## Description

This project develops an eye disease classification system from retinal fundus 
images, capable of distinguishing between four categories: **normal eye, cataract, 
diabetic retinopathy, and glaucoma**.

The system goes beyond training a classification model: it integrates a **full data 
engineering pipeline** (ingestion, validation, and versioning of data) and a 
**machine learning component** with experiment tracking, model explainability, and 
uncertainty quantification — so predictions are not just accurate, but interpretable 
and trustworthy.

The project is also developed within a **quality management framework** inspired by 
standards used for medical device software (ISO 13485, ISO 14971, IEC 62304), 
including document control, a traceability matrix, risk management, and an explicit 
separation between verification and validation — see `docs/`.

> ⚠️ Academic project. Not validated or certified for real clinical use.

## Motivation

Cataracts, diabetic retinopathy, and glaucoma are preventable causes of vision loss 
when detected early, but they require specialized evaluation that is not always 
available in time. An automated support system — always used under the supervision 
of a professional — can help prioritize cases and speed up screening.

## Dataset

- **Source:** [Eye Diseases Classification – Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Classes:** Normal, Cataract, Diabetic Retinopathy, Glaucoma
- **Image type:** retinal fundus photography
- **Size:** ~4,000+ public, anonymized images

## What makes this project different

Rather than stopping at a standard image classifier, this project focuses on the 
parts of a real ML system that most student projects skip:

| Area | What we implement |
|---|---|
| Explainability (XAI) | Grad-CAM / SHAP overlays showing which region of the retina drove each prediction |
| Uncertainty quantification | Monte Carlo Dropout or ensembling, instead of a raw softmax score |
| Out-of-distribution detection | Flags inputs that aren't valid fundus images, or are too low quality to evaluate |
| Risk-aware design | Every mitigation is traced back to a documented risk (see `docs/gestion_riesgos.md`) |

## Technical stack

| Area | Technologies |
|---|---|
| Data ingestion & pipeline | Python, pandas, Airflow/Prefect |
| Storage | PostgreSQL |
| Modeling | TensorFlow/PyTorch, transfer learning (EfficientNet/DenseNet) |
| Explainability | Grad-CAM, SHAP |
| Experiment tracking | MLflow, DVC |
| Infrastructure | Docker, docker-compose |

## Team

| Role | Member | Focus |
|---|---|---|
| Data Engineering | [Name] | ETL pipeline, data validation and storage |
| Machine Learning | [Name] | Training, evaluation, and model versioning |
| ML Reliability | [Name] | Explainability, uncertainty, out-of-distribution detection |

## Quality framework

See [`README-CALIDAD.md`](./README-CALIDAD.md) for the full detail of the document 
control system, risk management, and requirement → design → test → result traceability.

## Repository structure

```
├── README.md
├── README-CALIDAD.md
├── CHANGELOG.md
├── docs/
│   ├── requisitos.md
│   ├── diseno.md
│   ├── plan_de_pruebas.md
│   ├── matriz_trazabilidad.md
│   └── gestion_riesgos.md
├── data/           # (or .gitignore if data is large)
├── src/
└── notebooks/
```
