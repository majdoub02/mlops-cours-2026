import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("smoke-test")

with mlflow.start_run(run_name="hello-mlops"):
    mlflow.log_param("framework", "scikit-learn")
    mlflow.log_metric("accuracy", 0.42)
    mlflow.log_metric("loss", 1.337)
    print("Run logged successfully.")
