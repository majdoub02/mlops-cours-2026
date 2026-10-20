import os

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "titanic-classifier"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()

exp = client.get_experiment_by_name("titanic-classification")
best_runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.f1 DESC"],
    max_results=50,
)

run_to_version = {
    v.run_id: v.version for v in client.search_model_versions(f"name='{MODEL_NAME}'")
}

# On ne garde que les runs qui ont une version enregistrée
ranked = [(r, run_to_version[r.info.run_id]) for r in best_runs if r.info.run_id in run_to_version]

for i, (run, version) in enumerate(ranked[:3]):
    alias = "champion" if i == 0 else "staging"
    if alias == "staging" and i == 2:
        alias = "staging-2"  # un alias ne peut pointer que vers une version
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    client.set_model_version_tag(MODEL_NAME, version, "stage", "Production" if i == 0 else "Staging")
    print(f"Version {version} (f1={run.data.metrics['f1']:.4f}) -> {alias}")
