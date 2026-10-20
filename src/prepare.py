from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/titanic.csv")
PROC = Path("data/processed")
PARAMS = yaml.safe_load(open("params.yaml"))["prepare"]


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    df["Sex"] = (df["Sex"] == "female").astype(int)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S").map({"S": 0, "C": 1, "Q": 2})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "IsAlone"]
    X, y = df[features].fillna(0), df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PARAMS["test_size"], random_state=PARAMS["random_state"], stratify=y
    )
    X_train.to_parquet(PROC / "X_train.parquet")
    X_test.to_parquet(PROC / "X_test.parquet")
    y_train.to_frame().to_parquet(PROC / "y_train.parquet")
    y_test.to_frame().to_parquet(PROC / "y_test.parquet")
    print(f"Train: {len(X_train)} / Test: {len(X_test)}")


if __name__ == "__main__":
    main()
