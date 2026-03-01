import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_model(X, y):
    """
    Trains fraud detection model
    and logs experiment using MLflow.
    """

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Starting MLflow run...")

    with mlflow.start_run():

        # Model initialization
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        # Training
        model.fit(X_train, y_train)

        # Prediction probabilities
        probs = model.predict_proba(X_test)[:, 1]

        # ROC-AUC evaluation
        auc = roc_auc_score(y_test, probs)

        # Log parameters
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        # Log performance metric
        mlflow.log_metric("ROC_AUC", auc)

        # Save model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fraud_model"
        )

    print("\n✅ Training Finished Successfully")
    print(f"✅ ROC-AUC Score: {auc:.4f}")

    return model