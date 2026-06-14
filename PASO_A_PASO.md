# Laboratorio MLOps en AWS — Guía paso a paso

Reproducir el pipeline completo de extremo a extremo:
datos crudos → entrenamiento en AWS → API de inferencia → monitoreo.

> **Región:** `us-east-1` · **Cuenta:** `313694531227` · **Free Tier:** S3, DynamoDB, CodeBuild (100 min/mes), EC2 t3.micro

---

## Mapa del laboratorio

```
[datos CSV]
    │ preprocess → feature store
    ▼
[CodeBuild]  ←── buildspec.yml ejecuta el pipeline completo
    │ train → evaluate → register_model
    ▼
[S3]  modelo .joblib + métricas   [DynamoDB]  registro de versiones
    │                                  │
    └──────────── EC2 ────────────────┘
                   │  FastAPI + Docker
                   ▼
             /predict  /health
                   │
             [CloudWatch]  métricas de monitoreo
```

Cada bloque de comandos incluye la **salida esperada** para que sepas si está bien.

---

## 0. Antes de empezar

Verificá que tenés todo instalado y las credenciales configuradas:

```bash
aws --version
# AWS CLI 2.x.x

aws sts get-caller-identity
# {
#   "Account": "313694531227",
#   "UserId": "...",
#   "Arn": "arn:aws:iam::313694531227:user/..."
# }

python --version
# Python 3.11.x
```

Exportá estas variables una sola vez; se usan en todos los pasos siguientes:

```bash
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
export BUCKET=churn-mlops-$ACCOUNT_ID
echo "Bucket: $BUCKET"
# Bucket: churn-mlops-313694531227
```

> **Windows Git Bash:** si un argumento empieza con `/` (ARNs, rutas), anteponé
> `MSYS_NO_PATHCONV=1` al comando para que Git Bash no maniple las rutas.

---

## 1. Entorno local (opcional pero recomendado)

Antes de subir nada a AWS, confirmá que el pipeline corre bien en tu máquina.
Esto también ejecuta los tests y el linter.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt

export PYTHONPATH=src
python src/preprocess.py           # limpia el CSV crudo
python src/build_feature_store.py  # agrega timestamp, genera features.csv
python src/train.py                # entrena y guarda model.joblib + métricas
python src/evaluate.py             # quality gate: falla si f1 o AUC son bajos

ruff check .    # linter: sin errores
pytest          # 7 tests, todos verdes
```

Salida esperada al final de `pytest`:
```
7 passed in X.XXs
```

---

## 2. Crear la infraestructura base

### 2.1 Bucket S3

S3 es el almacén central del proyecto: guarda el dataset, el feature store, el
modelo entrenado, los experimentos de MLflow y los reportes de monitoreo.

```bash
aws s3api create-bucket --bucket $BUCKET --region $AWS_REGION

# Bloquear acceso público (buena práctica)
aws s3api put-public-access-block --bucket $BUCKET \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Subir el dataset crudo
aws s3 cp data/raw/telco_customer_churn_mlops.csv \
  s3://$BUCKET/raw/telco_customer_churn_mlops.csv

# Verificar
aws s3 ls s3://$BUCKET/
# 2026-...  telco_customer_churn_mlops.csv
```

### 2.2 Tabla DynamoDB (Model Registry)

DynamoDB actúa como registro de modelos: guarda la versión, las métricas y el
link al artefacto en S3. Reemplaza SageMaker Model Registry (que tiene cuotas
más estrictas).

```bash
aws dynamodb create-table \
  --table-name churn-model-registry \
  --attribute-definitions \
    AttributeName=model_name,AttributeType=S \
    AttributeName=version,AttributeType=S \
  --key-schema \
    AttributeName=model_name,KeyType=HASH \
    AttributeName=version,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region $AWS_REGION

# Salida esperada: "TableStatus": "CREATING" → en ~10 segundos pasa a ACTIVE
```

### 2.3 Repositorio ECR (imagen Docker)

ECR es el registry de imágenes Docker de AWS. La imagen de la API se construye
una vez y luego EC2 la descarga para correr el contenedor.

```bash
aws ecr create-repository --repository-name churn-api --region $AWS_REGION

# Salida esperada:
# "repositoryUri": "313694531227.dkr.ecr.us-east-1.amazonaws.com/churn-api"
```

---

## 3. Permisos IAM para CodeBuild

CodeBuild necesita permiso para escribir logs, leer/escribir en S3 y registrar
el modelo en DynamoDB. Creamos un rol específico con los mínimos permisos
necesarios (principio de menor privilegio).

```bash
# Política de confianza: quién puede asumir este rol
cat > /tmp/cb-trust.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "codebuild.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Permisos del rol
cat > /tmp/cb-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
      "Resource": "*"
    },
    {
      "Sid": "S3",
      "Effect": "Allow",
      "Action": ["s3:GetObject","s3:PutObject","s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::churn-mlops-$ACCOUNT_ID",
        "arn:aws:s3:::churn-mlops-$ACCOUNT_ID/*"
      ]
    },
    {
      "Sid": "DynamoDB",
      "Effect": "Allow",
      "Action": ["dynamodb:PutItem","dynamodb:GetItem","dynamodb:Query","dynamodb:DescribeTable","dynamodb:CreateTable"],
      "Resource": "arn:aws:dynamodb:$AWS_REGION:$ACCOUNT_ID:table/churn-model-registry"
    }
  ]
}
EOF

aws iam create-role \
  --role-name churn-codebuild-role \
  --assume-role-policy-document file:///tmp/cb-trust.json

aws iam put-role-policy \
  --role-name churn-codebuild-role \
  --policy-name churn-codebuild-policy \
  --policy-document file:///tmp/cb-policy.json
```

---

## 4. Proyecto CodeBuild (entrenamiento en la nube)

CodeBuild ejecuta `buildspec.yml`, que corre los 5 pasos del pipeline
(preprocess → feature store → train → evaluate → register_model) dentro de un
contenedor Linux gestionado por AWS. Reemplaza SageMaker Training Jobs.

### 4.1 Empaquetar y subir el código fuente a S3

```bash
git archive --format=zip HEAD -o /tmp/source.zip
aws s3 cp /tmp/source.zip s3://$BUCKET/source/source.zip

# Salida esperada:
# upload: /tmp/source.zip to s3://churn-mlops-313694531227/source/source.zip
```

### 4.2 Crear el proyecto CodeBuild

```bash
cat > /tmp/codebuild.json <<EOF
{
  "name": "churn-training",
  "source": {
    "type": "S3",
    "location": "churn-mlops-$ACCOUNT_ID/source/source.zip",
    "buildspec": "buildspec.yml"
  },
  "artifacts": { "type": "NO_ARTIFACTS" },
  "environment": {
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
    "computeType": "BUILD_GENERAL1_SMALL",
    "environmentVariables": [
      { "name": "S3_BUCKET",    "value": "churn-mlops-$ACCOUNT_ID" },
      { "name": "AWS_REGION",   "value": "us-east-1" }
    ]
  },
  "serviceRole": "arn:aws:iam::$ACCOUNT_ID:role/churn-codebuild-role",
  "timeoutInMinutes": 20
}
EOF

aws codebuild create-project \
  --cli-input-json file:///tmp/codebuild.json \
  --region $AWS_REGION
```

### 4.3 Lanzar el entrenamiento

```bash
BUILD_ID=$(aws codebuild start-build \
  --project-name churn-training \
  --query 'build.id' --output text)

echo "Build en curso: $BUILD_ID"

# Seguir el estado hasta que cambie a SUCCEEDED o FAILED
watch -n 15 "aws codebuild batch-get-builds \
  --ids '$BUILD_ID' \
  --query 'builds[0].buildStatus' --output text"

# Ver logs en tiempo real (Git Bash: prefijá MSYS_NO_PATHCONV=1)
MSYS_NO_PATHCONV=1 aws logs tail /aws/codebuild/churn-training --follow
```

Salida esperada al terminar:
```
SUCCEEDED
```

### 4.4 Verificar que el modelo quedó registrado

```bash
# Artefactos en S3
aws s3 ls s3://$BUCKET/models/ --recursive
# models/build-1/model.joblib
# models/build-1/metrics.json
# models/build-1/evaluation.json

# Entrada en DynamoDB
aws dynamodb scan --table-name churn-model-registry \
  --query 'Items[*].{version:version.S, status:status.S, f1:f1.N, auc:roc_auc.N}'
# [{"version":"build-1","status":"Production","f1":"0.7x","auc":"0.77xx"}]
```

> Si el build falla, revisá los logs de CloudWatch. El error más común es un
> permiso IAM faltante o un umbral de calidad demasiado alto en `evaluate.py`.

---

## 5. Permisos IAM para EC2

La instancia EC2 necesita descargar el modelo desde S3, consultarlo en DynamoDB
y publicar métricas en CloudWatch. También necesita push a ECR para construir la
imagen Docker durante el arranque (en producción, esto lo haría GitHub Actions).

```bash
cat > /tmp/ec2-trust.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "ec2.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

cat > /tmp/ec2-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::churn-mlops-$ACCOUNT_ID/*"
    },
    {
      "Effect": "Allow",
      "Action": ["dynamodb:Query","dynamodb:GetItem"],
      "Resource": "arn:aws:dynamodb:$AWS_REGION:$ACCOUNT_ID:table/churn-model-registry"
    },
    {
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData"],
      "Resource": "*"
    }
  ]
}
EOF

aws iam create-role \
  --role-name churn-ec2-role \
  --assume-role-policy-document file:///tmp/ec2-trust.json

aws iam put-role-policy \
  --role-name churn-ec2-role \
  --policy-name churn-ec2-policy \
  --policy-document file:///tmp/ec2-policy.json

# ECR PowerUser: permite que EC2 construya y pushee la imagen Docker
aws iam attach-role-policy \
  --role-name churn-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser

# Instance profile: es el "envoltorio" que asocia el rol a la instancia
aws iam create-instance-profile --instance-profile-name churn-ec2-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name churn-ec2-profile \
  --role-name churn-ec2-role
```

---

## 6. EC2 con FastAPI (endpoint de inferencia)

Lanzamos una instancia `t3.micro` (Free Tier). Al arrancar, el *user-data*
(script de inicialización) instala Docker, construye la imagen de la API a
partir del código en S3, la pushea a ECR y levanta el contenedor.

### 6.1 Security Group (firewall de la instancia)

Abrimos el puerto 8000 solo a tu IP pública para no exponer la API al mundo.

```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "Tu IP: $MY_IP"

SG_ID=$(aws ec2 create-security-group \
  --group-name churn-api-sg \
  --description "API de churn - solo mi IP" \
  --query 'GroupId' --output text)

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 8000 \
  --cidr $MY_IP/32

echo "Security Group: $SG_ID"
```

### 6.2 Script de arranque (user-data)

```bash
cat > /tmp/user-data.sh <<EOF
#!/bin/bash
set -xe
exec > /var/log/churn-setup.log 2>&1

ACCOUNT_ID=$ACCOUNT_ID
REGION=$AWS_REGION
BUCKET=churn-mlops-\$ACCOUNT_ID
ECR=\$ACCOUNT_ID.dkr.ecr.\$REGION.amazonaws.com/churn-api

dnf install -y docker unzip
systemctl enable --now docker

aws ecr get-login-password --region \$REGION \\
  | docker login --username AWS --password-stdin \\
    \$ACCOUNT_ID.dkr.ecr.\$REGION.amazonaws.com

aws s3 cp s3://\$BUCKET/source/source.zip /tmp/source.zip
mkdir -p /tmp/app && unzip -o /tmp/source.zip -d /tmp/app
cd /tmp/app

docker build -t \$ECR:latest .
docker push \$ECR:latest
docker run -d --restart always -p 8000:8000 \\
  -e AWS_REGION=\$REGION --name churn-api \$ECR:latest

echo "CHURN_API_READY"
EOF
```

### 6.3 Lanzar la instancia

```bash
# AMI de Amazon Linux 2023 (última versión, Free Tier)
AMI=$(aws ssm get-parameter \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameter.Value' --output text)

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI \
  --instance-type t3.micro \
  --iam-instance-profile Name=churn-ec2-profile \
  --security-group-ids $SG_ID \
  --user-data file:///tmp/user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=churn-api}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "Instancia: $INSTANCE_ID"

# Esperar hasta que esté corriendo
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Obtener la IP pública
IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "API en: http://$IP:8000"
```

La instancia tarda ~3 minutos en terminar de construir la imagen. Podés seguir
el progreso con:
```bash
MSYS_NO_PATHCONV=1 aws logs tail /aws/codebuild/churn-training --follow
```

### 6.4 Probar la API

```bash
# ¿Está viva?
curl http://$IP:8000/health
# {"status":"ok","model_type":"logistic-regression"}

# Predicción de un cliente de alto riesgo
curl -s -X POST http://$IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "contract_type": "Month-to-Month",
    "internet_service": "Fiber",
    "senior_citizen": 0,
    "tenure_months": 3,
    "monthly_charges": 95.0,
    "support_tickets_last_6m": 5
  }' | python -m json.tool
# {
#   "churn_prediction": 1,
#   "churn_probability": 0.8548,
#   "risk_label": "HIGH"
# }

# Predicción de un cliente de bajo riesgo
curl -s -X POST http://$IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "contract_type": "Two-Year",
    "internet_service": "DSL",
    "senior_citizen": 0,
    "tenure_months": 48,
    "monthly_charges": 45.0,
    "support_tickets_last_6m": 0
  }' | python -m json.tool
# {
#   "churn_prediction": 0,
#   "churn_probability": 0.051,
#   "risk_label": "LOW"
# }

# Swagger UI (documentación interactiva)
# Abrí en el navegador: http://$IP:8000/docs
```

---

## 7. Monitoreo (CloudWatch + S3)

`monitor.py` calcula métricas sobre las predicciones batch y las publica en
CloudWatch. Desde la consola podés ver gráficos y crear alertas.

```bash
# Inferencia sobre el feature store completo
PYTHONPATH=src python src/predict.py \
  --input data/feature_store/features.csv \
  --output reports/predictions.csv

# Publicar métricas y subir el reporte
PYTHONPATH=src python src/monitor.py \
  --predictions reports/predictions.csv

# Verificar que las métricas llegaron
aws cloudwatch list-metrics --namespace ChurnMLOps
# Deberías ver: PredictedChurnRate, AvgChurnProbability, HighRiskCustomers, TotalPredictions

# Ver el reporte en S3
aws s3 ls s3://$BUCKET/reports/
# 2026-... monitoring-YYYYMMDD-HHMMSS.json
```

En la consola AWS: **CloudWatch → Métricas → ChurnMLOps**.
Podés crear una alarma si `HighRiskCustomers` supera un umbral (señal de drift).

---

## 8. GitHub Actions (CI/CD automatizado)

Los workflows ya están listos en `.github/workflows/`. Para activarlos:

1. Subí el repo a GitHub.
2. Andá a **Settings → Secrets and variables → Actions** y agregá:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` → `us-east-1`

| Workflow | Cuándo corre | Qué hace |
|---|---|---|
| `ci.yml` | Cada push / PR | `ruff` + `pytest` (7 tests) |
| `train-codebuild.yml` | Manual desde Actions | Dispara CodeBuild y transmite logs |
| `docker-ecr.yml` | Push a `main` o manual | Build + push de la imagen a ECR |

> `train-codebuild.yml` requiere cambiar la fuente del proyecto CodeBuild a
> GitHub (en la consola: CodeBuild → churn-training → Edit → Source → GitHub).

---

## 9. Destruir los recursos (teardown)

Cuando termines el laboratorio, eliminá todo para no seguir gastando.
El único recurso que cobra continuamente es **EC2** (~$0.01/hora).

```bash
# 1. EC2 (lo primero, es lo que cobra)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID
aws ec2 delete-security-group --group-id $SG_ID

# 2. Roles IAM de EC2
aws iam remove-role-from-instance-profile \
  --instance-profile-name churn-ec2-profile --role-name churn-ec2-role
aws iam delete-instance-profile --instance-profile-name churn-ec2-profile
aws iam detach-role-policy \
  --role-name churn-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
aws iam delete-role-policy \
  --role-name churn-ec2-role --policy-name churn-ec2-policy
aws iam delete-role --role-name churn-ec2-role

# 3. CodeBuild + su rol
aws codebuild delete-project --name churn-training
aws iam delete-role-policy \
  --role-name churn-codebuild-role --policy-name churn-codebuild-policy
aws iam delete-role --role-name churn-codebuild-role

# 4. ECR, DynamoDB, S3
aws ecr delete-repository --repository-name churn-api --force
aws dynamodb delete-table --table-name churn-model-registry
aws s3 rb s3://$BUCKET --force

echo "Todos los recursos eliminados."
```

---

## Resumen del flujo

```
Paso 0  Verificar AWS CLI y credenciales
Paso 1  Correr el pipeline en local (opcional, para entender el código)
Paso 2  Crear S3 + DynamoDB + ECR
Paso 3  Crear rol IAM para CodeBuild
Paso 4  Crear proyecto CodeBuild y lanzar el entrenamiento
Paso 5  Crear rol IAM para EC2
Paso 6  Lanzar EC2 → la API queda disponible en http://<IP>:8000
Paso 7  Monitoreo con CloudWatch
Paso 8  (Opcional) Conectar GitHub Actions
Paso 9  Teardown
```
