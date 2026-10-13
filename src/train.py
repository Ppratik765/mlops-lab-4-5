import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

def main(n_estimators, max_depth):
    # Enable Autologging
    mlflow.sklearn.autolog()

    # Load data
    df = pd.read_csv("data/dataset.csv")
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("House_Price_Advanced")

    with mlflow.start_run() as run:
        # Train model
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate manually (Autolog handles training metrics, we log explicit test metrics)
        preds = model.predict(X_test)
        mlflow.log_metric("test_rmse", mean_squared_error(y_test, preds) ** 0.5)
        mlflow.log_metric("test_r2", r2_score(y_test, preds))

        # 1. Model Signature and Input Example
        signature = infer_signature(X_train, preds)
        input_example = X_train.head(3)

        # 2. Register Model to Model Registry (Updated for Skops security and deprecation warnings)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            signature=signature,
            input_example=input_example,
            registered_model_name="HousePriceModel",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print(f"Run complete! n_estimators={n_estimators}, max_depth={max_depth}")
        print(f"Logged to MLflow. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(estimators, depth)