# mlops-cours-2026

Labs du cours « Industrialisation et déploiement des modèles analytiques ».

| Dossier | Contenu |
|---|---|
| `lab0/` | Setup : cluster kind, MinIO, smoke test MLflow |
| `lab1-titanic/` | Pipeline DVC + MLflow (20+ runs) + Model Registry sur Titanic |

Chaque lab a son propre README (voir `lab1-titanic/README.md`).
Les données et artefacts sont stockés sur un MinIO local (kind), donc `dvc pull` ne fonctionne que sur ma machine.
Le dataset Titanic est public : https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
