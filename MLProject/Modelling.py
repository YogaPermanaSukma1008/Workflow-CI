import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from mlflow.models.signature import infer_signature

# ========== 1. Setup MLflow dengan DagsHub ==========
MLFLOW_URI = "https://dagshub.com/YogaPermanaSukma1008/membangun-model.mlflow"
mlflow_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

if not mlflow_username or not mlflow_password:
    raise ValueError("❌ MLFLOW credentials not found in environment.")

os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Default")

# Jangan aktifkan log_models di autolog karena DagsHub belum dukung model registry
mlflow.sklearn.autolog(log_models=False)

# ========== 2. Load Data ==========
try:
    X_train = pd.read_csv("MLProject/loandata_preprocessing/X_train_processed.csv")
    X_test = pd.read_csv("MLProject/loandata_preprocessing/X_test_processed.csv")
    y_train = pd.read_csv("MLProject/loandata_preprocessing/y_train.csv")
    y_test = pd.read_csv("MLProject/loandata_preprocessing/y_test.csv")

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
except Exception as e:
    raise FileNotFoundError(f"❌ Gagal memuat data: {e}")

# ========== 3. Logging Confusion Matrix ==========
def log_confusion_matrix(cm):
    try:
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
    except Exception as e:
        print(f"[ERROR] Gagal log confusion matrix: {e}")

# ========== 4. Logging ROC Curve ==========
def log_roc_curve(y_true, y_probs):
    try:
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
    except Exception as e:
        print(f"[ERROR] Gagal log ROC curve: {e}")

# ========== 5. Training dan Logging ==========
with mlflow.start_run(run_name="RandomForest_Default") as run:
    print(f"Run ID: {run.info.run_id}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, preds)

    # Logging metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Logging visual artifacts
    log_confusion_matrix(cm)
    log_roc_curve(y_test, probas)

    # Logging model (tanpa registered_model_name, agar tidak error)
    signature = infer_signature(X_test, preds)
    input_example = X_test.head(5)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # penting untuk mlflow artifacts download
        input_example=input_example,
        signature=signature
    )

print("✅ Model dan metrik berhasil dilog ke DagsHub.")