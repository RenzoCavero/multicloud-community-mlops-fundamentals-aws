# PASO A PASO — Laboratorio MLOps en AWS

Guía de laboratorio para levantar **todo el flujo** desde cero, ejecutarlo y al
final **destruir los recursos** para no gastar créditos.

- **Región:** `us-east-1`
- **Account ID (ejemplo):** `313694531227` → reemplázalo por el tuyo
- **Nombres de recursos** (todos parametrizables en `src/config.py`):
  - Bucket S3: `churn-mlops-<ACCOUNT_ID>`
  - Tabla DynamoDB: `churn-model-registry`
  - Repo ECR: `churn-api`
  - Proyecto CodeBuild: `churn-training`
  - Rol CodeBuild: `churn-codebuild-role` · Rol EC2: `churn-ec2-role`

> Todos los comandos usan **AWS CLI**. En Windows Git Bash, anteponé
> `MSYS_NO_PATHCONV=1` cuando un argumento empiece con `/` (ej. ARNs de logs).

---

## 0. Prerrequisitos

```bash
aws --version                 # AWS CLI v2
aws sts get-caller-identity   # confirma cuenta y credenciales
python --version              # 3.11
```

Exportá variables base (se usan en todos los pasos):

```bash
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1
export BUCKET=churn-mlops-$ACCOUNT_ID
echo "Bucket: $BUCKET"
```

> Si tu bucket no se llama `churn-mlops-<ACCOUNT_ID>`, editá `S3_BUCKET` en
> `src/config.py` o exportá `S3_BUCKET=<tu-bucket>`.

---

## 1. Entorno local (para entender y testear el flujo)

```bash
python -m venv .venv
source .venv/Scripts/activate     # Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt

# Pipeline completo en local (sin AWS)
export PYTHONPATH=src
python src/preprocess.py
python src/build_feature_store.py
python src/train.py
python src/evaluate.py

ruff check .
pytest
```

---

## 2. Crear la infraestructura base (S3, DynamoDB, ECR)

```bash
# S3 (datasets, feature store, modelos, mlruns, reportes)
aws s3api create-bucket --bucket $BUCKET --region $AWS_REGION
aws s3api put-public-access-block --bucket $BUCKET \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Dataset crudo a S3 (también queda versionado en el repo)
aws s3 cp data/raw/telco_customer_churn_mlops.csv s3://$BUCKET/raw/telco_customer_churn_mlops.csv

# DynamoDB (Model Registry)
aws dynamodb create-table --table-name churn-model-registry \
  --attribute-definitions AttributeName=model_name,AttributeType=S AttributeName=version,AttributeType=S \
  --key-schema AttributeName=model_name,KeyType=HASH AttributeName=version,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST --region $AWS_REGION

# ECR (imagen Docker de la API)
aws ecr create-repository --repository-name churn-api --region $AWS_REGION
```

---

## 3. Rol IAM para CodeBuild

CodeBuild necesita: escribir logs, leer/escribir el bucket y escribir en DynamoDB.

`cb-trust.json`:
```json
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"codebuild.amazonaws.com"},"Action":"sts:AssumeRole"}]}
```

`cb-policy.json` (reemplazá `<ACCOUNT_ID>`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid":"Logs","Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"},
    {"Sid":"S3","Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],"Resource":["arn:aws:s3:::churn-mlops-<ACCOUNT_ID>","arn:aws:s3:::churn-mlops-<ACCOUNT_ID>/*"]},
    {"Sid":"DynamoDB","Effect":"Allow","Action":["dynamodb:PutItem","dynamodb:GetItem","dynamodb:Query","dynamodb:DescribeTable","dynamodb:CreateTable"],"Resource":["arn:aws:dynamodb:us-east-1:<ACCOUNT_ID>:table/churn-model-registry","arn:aws:dynamodb:us-east-1:<ACCOUNT_ID>:table/churn-model-registry/*"]}
  ]
}
```

```bash
aws iam create-role --role-name churn-codebuild-role \
  --assume-role-policy-document file://cb-trust.json
aws iam put-role-policy --role-name churn-codebuild-role \
  --policy-name churn-codebuild-policy --policy-document file://cb-policy.json
```

---

## 4. Proyecto CodeBuild = entrenamiento en AWS

CodeBuild ejecuta `buildspec.yml` (preprocess → feature store → train → evaluate →
register). Hay dos formas de darle el código fuente:

### Opción A — Fuente en S3 (rápida, sin GitHub; ideal para probar ya)

```bash
# Empaquetar el repo (solo archivos versionados) y subirlo
git archive --format=zip HEAD -o source.zip
aws s3 cp source.zip s3://$BUCKET/source/source.zip && rm source.zip
```

`codebuild.json` (reemplazá `<ACCOUNT_ID>`):
```json
{
  "name": "churn-training",
  "source": { "type": "S3", "location": "churn-mlops-<ACCOUNT_ID>/source/source.zip", "buildspec": "buildspec.yml" },
  "artifacts": { "type": "NO_ARTIFACTS" },
  "environment": {
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
    "computeType": "BUILD_GENERAL1_SMALL",
    "environmentVariables": [
      { "name": "S3_BUCKET", "value": "churn-mlops-<ACCOUNT_ID>" },
      { "name": "AWS_REGION", "value": "us-east-1" }
    ]
  },
  "serviceRole": "arn:aws:iam::<ACCOUNT_ID>:role/churn-codebuild-role",
  "timeoutInMinutes": 20
}
```

```bash
aws codebuild create-project --cli-input-json file://codebuild.json --region $AWS_REGION

# Lanzar el entrenamiento y seguir el estado
BUILD_ID=$(aws codebuild start-build --project-name churn-training --query 'build.id' --output text)
aws codebuild batch-get-builds --ids "$BUILD_ID" --query 'builds[0].buildStatus'
```

> Cuando cambies el código, repetí el `git archive` + `aws s3 cp` y volvé a lanzar
> el build (la fuente S3 no se actualiza sola).

### Opción B — Fuente en GitHub (para el trigger de GitHub Actions)

1. Subí el repo a GitHub.
2. En la consola: **CodeBuild → Settings → Connections → GitHub** (autorización OAuth, una sola vez).
3. Recreá el proyecto con `"source": {"type": "GITHUB", "location": "https://github.com/<user>/<repo>.git", "buildspec": "buildspec.yml"}`.
4. El workflow `train-codebuild.yml` lo dispara desde **Actions** (ver paso 8).

### Verificar el resultado del entrenamiento

```bash
# Artefactos del modelo en S3
aws s3 ls s3://$BUCKET/models/ --recursive
# Registry en DynamoDB (versiones del modelo)
aws dynamodb scan --table-name churn-model-registry --query 'Items'
# Historial de experimentos MLflow
aws s3 ls s3://$BUCKET/mlruns/ --recursive | head
```

Ver logs del build (Git Bash necesita `MSYS_NO_PATHCONV=1`):
```bash
MSYS_NO_PATHCONV=1 aws logs tail /aws/codebuild/churn-training --follow
```

---

## 5. Rol IAM e imagen para EC2

### 5.1 Rol de instancia EC2

`ec2-trust.json`:
```json
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
```

`ec2-policy.json` (lectura de S3, query a DynamoDB, métricas CloudWatch):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect":"Allow","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::churn-mlops-<ACCOUNT_ID>/*"},
    {"Effect":"Allow","Action":["dynamodb:Query","dynamodb:GetItem"],"Resource":"arn:aws:dynamodb:us-east-1:<ACCOUNT_ID>:table/churn-model-registry"},
    {"Effect":"Allow","Action":["cloudwatch:PutMetricData"],"Resource":"*"}
  ]
}
```

```bash
aws iam create-role --role-name churn-ec2-role --assume-role-policy-document file://ec2-trust.json
aws iam put-role-policy --role-name churn-ec2-role --policy-name churn-ec2-policy --policy-document file://ec2-policy.json
# ECR: para CONSTRUIR/PUSHEAR desde EC2 en la demo usamos PowerUser; en el flujo
# normal (la imagen la sube GitHub Actions) basta ReadOnly.
aws iam attach-role-policy --role-name churn-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
aws iam create-instance-profile --instance-profile-name churn-ec2-profile
aws iam add-role-to-instance-profile --instance-profile-name churn-ec2-profile --role-name churn-ec2-role
```

### 5.2 Imagen Docker de la API

**Flujo normal:** la sube `docker-ecr.yml` desde GitHub Actions (paso 8).
**En la demo** (sin GitHub) la construye la propia EC2 vía *user-data* (paso 6).

---

## 6. EC2 + Docker + FastAPI (endpoint de inferencia)

> Los comandos exactos y verificados se completan en la sección final del lab.
> La idea: lanzar una `t3.micro` con un *security group* que abra el puerto 8000
> y un *user-data* que instale Docker, construya/baje la imagen y corra el contenedor.

```bash
# Security group con puerto 8000 abierto (demo)
SG_ID=$(aws ec2 create-security-group --group-name churn-api-sg \
  --description "Churn API" --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

`user-data.sh` (instala Docker, construye la imagen y corre el contenedor):
```bash
#!/bin/bash
dnf install -y docker && systemctl start docker
ACCOUNT_ID=<ACCOUNT_ID>; REGION=us-east-1; BUCKET=churn-mlops-$ACCOUNT_ID
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
aws s3 cp s3://$BUCKET/source/source.zip /tmp/source.zip
cd /tmp && unzip -o source.zip -d app && cd app
docker build -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/churn-api:latest .
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/churn-api:latest
docker run -d -p 8000:8000 -e AWS_REGION=$REGION $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/churn-api:latest
```

```bash
# Lanzar la instancia (AMI Amazon Linux 2023, Free Tier t3.micro)
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query 'Parameter.Value' --output text)
INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI --instance-type t3.micro \
  --iam-instance-profile Name=churn-ec2-profile --security-group-ids $SG_ID \
  --user-data file://user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=churn-api}]' \
  --query 'Instances[0].InstanceId' --output text)

# IP pública
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

### Probar la API

```bash
IP=<public-ip>
curl http://$IP:8000/health
curl -X POST http://$IP:8000/predict -H "Content-Type: application/json" -d '{
  "gender":"Male","contract_type":"Month-to-Month","internet_service":"Fiber",
  "senior_citizen":0,"tenure_months":3,"monthly_charges":95.0,"support_tickets_last_6m":5}'
# -> {"churn_prediction":1,"churn_probability":0.7x,"risk_label":"HIGH"}
```

---

## 7. Monitoreo (CloudWatch + S3)

```bash
# Inferencia batch sobre un CSV de clientes
PYTHONPATH=src python src/predict.py --input data/feature_store/features.csv --output reports/predictions.csv
# Publicar métricas a CloudWatch y subir el reporte a S3
PYTHONPATH=src python src/monitor.py --predictions reports/predictions.csv

# Ver métricas
aws cloudwatch list-metrics --namespace ChurnMLOps
aws s3 ls s3://$BUCKET/reports/
```

En la consola: **CloudWatch → Metrics → ChurnMLOps** (PredictedChurnRate,
AvgChurnProbability, HighRiskCustomers). Podés crear una **alarma** si, por
ejemplo, `HighRiskCustomers` supera un umbral (señal de posible drift).

---

## 8. GitHub Actions (CI + triggers)

En GitHub → **Settings → Secrets and variables → Actions**, agregá:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.

| Workflow | Cuándo corre | Qué hace | Semáforo |
|---|---|---|---|
| `ci.yml` | cada push / PR | `ruff` + `pytest` | check verde/rojo + badge en README |
| `train-codebuild.yml` | manual (Actions) | dispara CodeBuild y transmite logs | check + link al build |
| `docker-ecr.yml` | push a `main` o manual | build + push de la imagen a ECR | check verde/rojo |

> `train-codebuild.yml` requiere el proyecto CodeBuild con **fuente GitHub**
> (paso 4, opción B). `docker-ecr.yml` requiere el repo ECR (paso 2).

---

## 9. Destrucción de recursos (teardown)

```bash
# EC2
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 delete-security-group --group-id $SG_ID          # tras la terminación

# IAM EC2
aws iam remove-role-from-instance-profile --instance-profile-name churn-ec2-profile --role-name churn-ec2-role
aws iam delete-instance-profile --instance-profile-name churn-ec2-profile
aws iam detach-role-policy --role-name churn-ec2-role --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
aws iam delete-role-policy --role-name churn-ec2-role --policy-name churn-ec2-policy
aws iam delete-role --role-name churn-ec2-role

# CodeBuild + IAM
aws codebuild delete-project --name churn-training
aws iam delete-role-policy --role-name churn-codebuild-role --policy-name churn-codebuild-policy
aws iam delete-role --role-name churn-codebuild-role

# ECR / DynamoDB / S3
aws ecr delete-repository --repository-name churn-api --force
aws dynamodb delete-table --table-name churn-model-registry
aws s3 rb s3://$BUCKET --force
```

> Lo único que cuesta de forma continua es **EC2**: si no la usás, hacé
> `terminate-instances` (o `stop-instances` para conservarla apagada).
