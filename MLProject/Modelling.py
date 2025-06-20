import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from mlflow.models.signature import infer_signature

# ======== 0. Argument Parsing ========
parser = argparse.ArgumentParser()
parser.add_argument("--x_train_path", type=str, required=True)
parser.add_argument("--x_test_path", type=str, required=True)
parser.add_argument("--y_train_path", type=str, required=True)
parser.add_argument("--y_test_path", type=str, required=True)
parser.add_argument("--model_output", type=str, default="output/model.pkl")
args = parser.parse_args()


# ======== 1. Setup Dual Tracking ========
dagshub_username = os.getenv("DAGSHUB_USERNAME", "")
dagshub_token = os.getenv("DAGSHUB_TOKEN", "")

local_uri = "file:///" + os.path.abspath("./mlruns")
os.makedirs("./mlruns", exist_ok=True)
mlflow.set_tracking_uri(local_uri)
print(f"✅ Local tracking: mlruns active at {local_uri}")

if dagshub_username and dagshub_token:
    dagshub_uri = f"https://dagshub.com/{dagshub_username}/ml_flow.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    print("✅ Remote tracking: DagsHub active")
else:
    dagshub_uri = None
    print("⚠️ Remote tracking: DagsHub credentials not found")

# ======== 2. Load Data ========
X_train = pd.read_csv("loandata_preprocessing/X_train_processed.csv")
X_test = pd.read_csv("loandata_preprocessing/X_test_processed.csv")
y_train = pd.read_csv("loandata_preprocessing/y_train.csv").values.ravel()
y_test = pd.read_csv("loandata_preprocessing/y_test.csv").values.ravel()

print("🔎 Data Shape Info")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", pd.Series(y_train).value_counts())
print("y_test:", pd.Series(y_test).value_counts())

# ======== 3. Visual Logging ========
def save_and_log_plot(plot_func, filename):
    folder = os.path.join("output", "artifacts")
    os.makedirs(folder, exist_ok=True)
    fig_path = os.path.join(folder, filename)
    plot_func(fig_path)
    return fig_path

def create_confusion_matrix(cm, fig_path):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def create_roc_curve(y_true, y_probs, fig_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

# ======== 4. Dual Logging Implementation ========
def dual_logging(metrics, artifacts, model=None, X_sample=None):
    # --- LOCAL LOGGING ---
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="Local_RF", nested=True) as local_run:
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for art in artifacts:
            if os.path.exists(art):
                mlflow.log_artifact(art)
            else:
                print(f"⚠️ Artifact not found, skipping: {art}")
        if model and X_sample is not None:
            signature = infer_signature(X_sample, model.predict(X_sample))
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=X_sample[:5],
                signature=signature
            )
        print(f"📡 Logged to local MLflow run: {local_run.info.run_id}")

    # --- REMOTE DAGSHUB LOGGING ---
    if dagshub_uri:
        mlflow.set_tracking_uri(dagshub_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="Remote_RF") as remote_run:
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for art in artifacts:
                if os.path.exists(art):
                    mlflow.log_artifact(art)
                else:
                    print(f"⚠️ Artifact not found, skipping: {art}")
            if model:
                # Simpan model manual (tidak pakai log_model karena error di DagsHub)
                model_file = "model.joblib"
                joblib.dump(model, model_file)
                mlflow.log_artifact(model_file, artifact_path="model_artifact")
            print(f"🌐 Logged to DagsHub run: {remote_run.info.run_id}")
    else:
        print("⚠️ Skipping DagsHub logging (no credentials)")

# ======== 5. MLflow Run ========
experiment_name = "RandomForest_DualTracking"
mlflow.set_tracking_uri(local_uri)
mlflow.set_experiment(experiment_name)

print("🚀 Starting model training...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and Metrics
preds = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]
metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds, zero_division=0),
    "recall": recall_score(y_test, preds, zero_division=0),
    "f1": f1_score(y_test, preds, zero_division=0),
    "roc_auc": roc_auc_score(y_test, probas)
}

# Visualizations
cm = confusion_matrix(y_test, preds)
cm_path = save_and_log_plot(lambda p: create_confusion_matrix(cm, p), "confusion_matrix.png")
roc_path = save_and_log_plot(lambda p: create_roc_curve(y_test, probas, p), "roc_curve.png")

# Save model locally (backup)
os.makedirs("output", exist_ok=True)
model_path = os.path.join("output", "model.pkl")
joblib.dump(model, model_path)
print(f"✅ Model saved to {model_path}")

# Dual Tracking Log
dual_logging(metrics, [cm_path, roc_path, model_path], model=model, X_sample=X_test)

print("✅ Tracking completed on both local mlruns and DagsHub (if credentials available)")
