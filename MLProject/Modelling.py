import os
import tempfile
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
import joblib

# ======== 1. Setup MLflow Tracking ========
if os.getenv("DAGSHUB_USERNAME") and os.getenv("DAGSHUB_TOKEN"):
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.environ.get("DAGSHUB_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.environ.get("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri("https://dagshub.com/YogaPermanaSukma1008/membangun-model.mlflow")
    mlflow.set_experiment("RandomForest_Default")
    remote_tracking = True
    print("✅ Using remote MLflow tracking via DagsHub.")
else:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("RandomForest_Default")
    remote_tracking = False
    print("⚠️ Using local MLflow tracking.")

mlflow.sklearn.autolog(log_models=True)

# ======== 2. Load Data ========
X_train = pd.read_csv("loandata_preprocessing/X_train_processed.csv")
X_test = pd.read_csv("loandata_preprocessing/X_test_processed.csv")
y_train = pd.read_csv("loandata_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("loandata_preprocessing/y_test.csv").values.ravel()

# ======== 3. Visual Logging ========
def log_confusion_matrix(cm):
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_path = os.path.join(tmp_dir, "confusion_matrix.png")
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path, artifact_path="confusion_matrix")

def log_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig_path = os.path.join(tmp_dir, "roc_curve.png")
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path, artifact_path="roc_curve")

# ======== 4. MLflow Run ========
with mlflow.start_run(run_name="RandomForest_Classifier") as run:
    print(f"🚀 MLflow run ID: {run.info.run_id}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, preds)

    # Manual metrics (in addition to autolog)
    mlflow.log_metric("manual_accuracy", acc)
    mlflow.log_metric("manual_precision", prec)
    mlflow.log_metric("manual_recall", rec)
    mlflow.log_metric("manual_f1_score", f1)
    mlflow.log_metric("manual_roc_auc", roc_auc)

    log_confusion_matrix(cm)
    log_roc_curve(y_test, probas)

    # Model Logging (again, to register manually if needed)
    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_test.head())

    # Save model locally
    os.makedirs("output", exist_ok=True)
    joblib.dump(model, "output/model.pkl")
    mlflow.log_artifact("output/model.pkl")

    # Model Registry (if remote)
    if remote_tracking:
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_model_name = "rf-loan-classifier"
        try:
            mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"✅ Model registered as '{registered_model_name}'")
        except Exception as e:
            print(f"❌ Failed to register model: {e}")

    print("✅ Tracking selesai.")
