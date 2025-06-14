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

# ========== 1. Setup MLflow dengan DagsHub ==========
# Ambil kredensial dari environment variable (disediakan oleh GitHub Actions)
mlflow_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

# Set kredensial ke environment (jika dibutuhkan oleh MLflow secara eksplisit)
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

mlflow.set_tracking_uri("https://dagshub.com/YogaPermanaSukma1008/membangun-model.mlflow")

# Aktifkan autolog
mlflow.sklearn.autolog(log_models=True)

# ========== 2. Load Data ==========
X_train = pd.read_csv("loandata_preprocessing/X_train_processed.csv")
X_test = pd.read_csv("loandata_preprocessing/X_test_processed.csv")
y_train = pd.read_csv("loandata_preprocessing/y_train.csv")
y_test = pd.read_csv("loandata_preprocessing/y_test.csv")

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# ========== 3. Fungsi logging confusion matrix ==========
def log_confusion_matrix(cm):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = os.path.join(tmp_dir, "confusion_matrix.png")
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            mlflow.log_artifact(fig_path, artifact_path="confusion_matrix")
    except Exception as e:
        print(f"[ERROR] Gagal log confusion matrix: {e}")

# ========== 4. Fungsi logging ROC Curve ==========
def log_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig_path = os.path.join(tmp_dir, "roc_curve.png")
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            mlflow.log_artifact(fig_path, artifact_path="roc_curve")
    except Exception as e:
        print(f"[ERROR] Gagal log ROC curve: {e}")

# ========== 5. Mulai MLflow run ==========
with mlflow.start_run(run_name="RandomForest_Default") as run:
    print(f"Run ID: {run.info.run_id}")  # Debug run ID

    # Inisialisasi dan training model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]  # Probabilitas untuk kelas 1

    # Evaluasi
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='binary', zero_division=0)
    rec = recall_score(y_test, preds, average='binary', zero_division=0)
    f1 = f1_score(y_test, preds, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, preds)

    # Logging manual (melengkapi autolog)
    mlflow.log_metric("manual_accuracy", acc)
    mlflow.log_metric("manual_precision", prec)
    mlflow.log_metric("manual_recall", rec)
    mlflow.log_metric("manual_f1_score", f1)
    mlflow.log_metric("manual_roc_auc", roc_auc)

        # Logging manual (melengkapi autolog)
    mlflow.log_metric("manual_accuracy", acc)
    mlflow.log_metric("manual_precision", prec)
    mlflow.log_metric("manual_recall", rec)
    mlflow.log_metric("manual_f1_score", f1)
    mlflow.log_metric("manual_roc_auc", roc_auc)

    # Logging eksplisit (untuk ditampilkan di MLflow dashboard)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Log artefak visual
    log_confusion_matrix(cm)
    log_roc_curve(y_test, probas)

    # Logging eksplisit model
    mlflow.sklearn.log_model(model, artifact_path = "model")

print("✅ Model dan metrik berhasil dilog ke DagsHub.")
