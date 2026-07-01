# multicloud-community-mlops-fundamentals-aws

[![CI](https://github.com/renzocavero/multicloud-community-mlops-fundamentals-aws/actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

Flujo **MLOps end-to-end y didáctico** para predecir *churn* (abandono) de clientes
de una fintech, usando **AWS real** pero sin sobreingeniería. Pensado para una clase:
pocos scripts, bien nombrados, fáciles de explicar.

> El entrenamiento **corre en AWS** (CodeBuild), no en local. No se usa SageMaker
> (Training Jobs / Feature Store / Model Registry) porque la cuenta tiene esas
> cuotas restringidas; se reemplazan con servicios equivalentes y simples.

---

## 1. Caso de negocio

Una fintech quiere **anticipar qué clientes van a abandonar** para activar acciones
preventivas (descuentos, contacto proactivo, mejoras de servicio). El modelo recibe
los datos de un cliente y devuelve:

- `churn_prediction`: 0 (se queda) / 1 (se va)
- `churn_probability`: probabilidad de churn [0, 1]
- `risk_label`: **LOW / MEDIUM / HIGH** (para priorizar acciones)

Dataset: `data/raw/telco_customer_churn_mlops.csv` (300 clientes).

| Columna | Rol |
|---|---|
| `customer_id` | identificador (se descarta para el modelo, se guarda en el feature store) |
| `gender`, `contract_type`, `internet_service` | features categóricas |
| `senior_citizen`, `tenure_months`, `monthly_charges`, `support_tickets_last_6m` | features numéricas |
| `churn` | target (Yes/No → 1/0) |

---

## 2. Arquitectura

```
GitHub Actions ──► AWS CodeBuild ──► S3 ──► DynamoDB (Model Registry)
   (CI + trigger)   (entrenamiento)  (artefactos)        │
        │                                                ▼
        └────► ECR (imagen API) ──► EC2 + Docker + FastAPI ──► CloudWatch / S3
                                          (inferencia)            (monitoreo)
```

**Pipeline de entrenamiento (en CodeBuild, definido en `buildspec.yml`):**

```
preprocess.py ─► build_feature_store.py ─► train.py ─► evaluate.py ─► register_model.py
 (limpia CSV)      (feature store CSV→S3)   (MLflow)   (quality gate)   (S3 + DynamoDB)
```

**Pipeline de inferencia:**

```
EC2 (Docker/FastAPI) ─► inference.py lee la versión Production del registry
                        (DynamoDB) ─► descarga model.joblib de S3 ─► /predict
```

---

## 3. Servicios AWS: qué hace cada uno y por qué

| Servicio | Para qué lo usamos | Por qué esta decisión |
|---|---|---|
| **S3** | Datasets, feature store offline (CSV), modelos, métricas, metadata, runs de MLflow y reportes de monitoreo | Almacenamiento barato y universal; es el "disco" de todo el flujo. Free Tier: 5 GB. |
| **CodeBuild** | Ejecuta el entrenamiento en AWS (reemplaza SageMaker Training Job) | Da un contenedor Linux administrado donde corre `pip install` + el pipeline. Sin cuotas de SageMaker. Free Tier: 100 min/mes. |
| **DynamoDB** | Model Registry: una fila por versión de modelo con métricas, URI en S3 y `status` | Registry serverless simple; reemplaza SageMaker Model Registry. Free Tier: 25 GB. |
| **ECR** | Guarda la imagen Docker de la API | Registry de contenedores nativo, se integra con ECR login + EC2. Free Tier: 500 MB. |
| **EC2** | Sirve la API FastAPI con Docker (endpoint de inferencia) | Una `t2.micro`/`t3.micro` Free Tier basta para la demo; control total y barato. |
| **CloudWatch** | Logs del build/API y métricas de monitoreo (tasa de churn, % HIGH, etc.) | Observabilidad nativa; permite dashboards y alarmas. Free Tier generoso. |
| **IAM** | Roles mínimos para CodeBuild y EC2 | Principio de menor privilegio; sin claves hardcodeadas. |
| **MLflow** | Tracking del experimento *dentro* del job de CodeBuild (file store → S3) | Registra params/métricas/modelo sin montar un servidor 24/7. El historial queda en S3. |

**Por qué un feature store en CSV y no SageMaker Feature Store:** la cuenta tiene
cuotas restringidas y, para este caso, un CSV versionado en S3 cumple el mismo rol
didáctico (tabla de features con identificador y `event_timestamp`, lista para
entrenar e inferir) sin costo ni complejidad extra.

---

## 4. Tecnologías y setup mínimo

- **Python 3.11**, `scikit-learn` (modelo), `pandas`, `mlflow`, `joblib`
- **FastAPI + uvicorn** (API), **Docker** (empaquetado), **boto3** (AWS)
- **GitHub Actions** (CI + triggers), **AWS CLI** (provisión)
- Lint **ruff**, tests **pytest**

```bash
# Entorno local
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; en Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt

# Pipeline completo en local (sin AWS), útil para entender el flujo
export PYTHONPATH=src
python src/preprocess.py
python src/build_feature_store.py
python src/train.py
python src/evaluate.py

# API local
PYTHONPATH=src uvicorn api:app --reload --port 8000   # http://localhost:8000/docs

# Calidad
ruff check .
pytest
```

> **Para desplegar en AWS paso a paso, ver [PASO_A_PASO.md](PASO_A_PASO.md).**

---

## Estructura del repo

```
src/
  config.py              # constantes: features, nombres de recursos AWS, umbrales
  preprocess.py          # 1) limpia el CSV crudo
  build_feature_store.py # 2) construye el feature store offline (CSV → S3)
  train.py               # 3) entrena + MLflow (file store)
  evaluate.py            # 4) evalúa + quality gate
  inference.py           # carga del modelo (local o desde el registry) + predicción
  predict.py             # pipeline de inferencia batch (CSV)
  api.py                 # API FastAPI (/health, /model/info, /predict, /predict/batch)
  monitor.py             # monitoreo: métricas a CloudWatch + reporte a S3
infra/
  register_model.py      # 5) registra el modelo en S3 + DynamoDB
buildspec.yml            # pipeline de entrenamiento para CodeBuild
Dockerfile               # imagen de la API
.github/workflows/       # ci.yml, train-codebuild.yml, docker-ecr.yml
tests/                   # pytest
data/raw/                # dataset
```

---

## Decisiones de diseño (resumen)

- **Sin SageMaker** (cuotas): CodeBuild = training, S3+DynamoDB = registry, CSV en S3 = feature store.
- **Scripts planos con `argparse`**, `import config` (src es la raíz de fuentes), rutas como constantes claras.
- **Quality gate** en `evaluate.py`: si f1/AUC no superan el umbral, el build falla y *no* se registra el modelo.
- **El mismo código de inferencia** sirve en local (`MODEL_PATH`) y en producción (descarga la versión `Production` del registry).
- **Sin secretos en el repo**: credenciales vía GitHub Secrets / roles IAM de EC2 y CodeBuild.
