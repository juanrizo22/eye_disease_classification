# Eye Disease Classification — Sistema de apoyo diagnóstico oftalmológico

## Descripción

Este proyecto desarrolla un sistema de clasificación de enfermedades oculares a partir 
de imágenes de fondo de ojo (retinal fundus images), capaz de distinguir entre cuatro 
categorías: **ojo normal, catarata, retinopatía diabética y glaucoma**.

El sistema no se limita a entrenar un modelo de clasificación: integra un **pipeline 
completo de ingeniería de datos** (ingesta, validación y versionado de datos), un 
**componente de machine learning** con tracking de experimentos, y un **dashboard 
interactivo** para visualización de resultados y nivel de confianza de las predicciones.

Adicionalmente, el proyecto se desarrolla dentro de un **marco de gestión de calidad** 
inspirado en normas de software para dispositivos médicos (ISO 13485, ISO 14971, 
IEC 62304), incluyendo control documental, matriz de trazabilidad, gestión de riesgos 
y separación explícita entre verificación y validación — ver `docs/`.

> ⚠️ Proyecto académico. No está validado ni certificado para uso clínico real.

## Motivación

La catarata, la retinopatía diabética y el glaucoma son causas prevenibles de pérdida 
de visión cuando se detectan a tiempo, pero requieren evaluación especializada que no 
siempre está disponible de forma oportuna. Un sistema de apoyo automatizado —usado 
siempre bajo supervisión de un profesional— puede ayudar a priorizar casos y agilizar 
el tamizaje.

## Dataset

- **Fuente:** [Eye Diseases Classification – Kaggle](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Clases:** Normal, Catarata, Retinopatía Diabética, Glaucoma
- **Tipo de imagen:** fondo de ojo (retinal fundus)
- **Tamaño:** ~4,000+ imágenes públicas y anonimizadas

## Stack técnico

| Área | Tecnologías |
|---|---|
| Ingesta y pipeline de datos | Python, pandas, Airflow/Prefect |
| Almacenamiento | PostgreSQL |
| Modelado | TensorFlow/PyTorch, transfer learning (EfficientNet/DenseNet) |
| Tracking de experimentos | MLflow, DVC |
| Visualización | Streamlit/Dash |
| Infraestructura | Docker, docker-compose |

## Equipo

| Rol | Integrante | Enfoque |
|---|---|---|
| Data Engineering | [Nombre] | Pipeline ETL, validación y almacenamiento de datos |
| Machine Learning | [Nombre] | Entrenamiento, evaluación y versionado del modelo |
| Visualización / Producto | [Nombre] | Dashboard, métricas y experiencia de usuario |

## Marco de calidad

Ver [`README-CALIDAD.md`](./README-CALIDAD.md) para el detalle completo del sistema 
de control documental, gestión de riesgos y trazabilidad requisito → diseño → prueba → resultado.

## Estructura del repositorio

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
├── data/           # (o .gitignore si son datos pesados)
├── src/
└── notebooks/
```
