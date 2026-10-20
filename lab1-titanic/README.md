# m1-titanic : projet ML versionné (DVC + MLflow + MinIO)

## Prérequis
- Cluster kind avec MinIO, port-forward actif : `kubectl -n minio port-forward svc/minio 9000:9000`
- Buckets `mlops-lab1` (DVC) et `mlflow-artifacts` (MLflow)
- Python 3.11 et uv

## Reproduire
```bash
uv sync
source .venv/bin/activate
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin123
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 MLFLOW_TRACKING_URI=http://localhost:5000
dvc pull
dvc repro
```

## Contenu
- `src/prepare.py`, `src/train.py`, `src/grid_search.py`, `src/promote.py`
- `dvc.yaml` / `dvc.lock` : pipeline prepare -> train
- `params.yaml` : hyperparamètres

## Registry MLflow
MLflow 3.x a déprécié les stages. Équivalences utilisées :
- alias `champion` = Production
- alias `staging` et `staging-2` = Staging (un alias ne désigne qu'une version)
- tag `stage` = Production / Staging sur chaque version concernée
Chargement : `models:/titanic-classifier@champion`
