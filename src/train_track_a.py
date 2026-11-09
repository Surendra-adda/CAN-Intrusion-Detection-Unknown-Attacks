
import pandas as pd
import numpy as np
import time

from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
import joblib

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "processed_data"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

FEATURES = (
    ["can_id", "payload_length"]
    + [f"byte_{i}" for i in range(8)]
    + [
        "delta_t_ms",
        "same_id_delta_t_ms",
        "rolling_mean_delta_t_ms"
    ]
)

RANDOM_STATE = 42
CALIBRATION_QUANTILE = 0.99

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

train = pd.read_csv(DATA / "train.csv")
calibration = pd.read_csv(DATA / "calibration.csv")
test = pd.read_csv(DATA / "unknown_test.csv")

# Train only on benign traffic
X_train = train.loc[
    train["label"] == "BENIGN", FEATURES
].copy()

# Calibrate using only benign traffic
X_calibration = calibration.loc[
    calibration["label"] == "BENIGN", FEATURES
].copy()

X_test = test[FEATURES].copy()

# Binary test labels:
# 0 = benign, 1 = unknown attack
y_test = (
    test["label"] != "BENIGN"
).astype(int).to_numpy()

print("\nDataset sizes")
print("Training benign:", len(X_train))
print("Calibration benign:", len(X_calibration))
print("Test records:", len(X_test))
print("Unknown attacks:", int(y_test.sum()))
print("Benign test:", int((y_test == 0).sum()))

# --------------------------------------------------
# TRAIN ISOLATION FOREST
# --------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=300,
    max_samples=2048,
    contamination="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

start = time.perf_counter()

model.fit(X_train)

training_time = time.perf_counter() - start

print(f"Training time: {training_time:.2f} seconds")

# --------------------------------------------------
# CALIBRATE THRESHOLD
# --------------------------------------------------

print("\nCalibrating threshold using benign data...")

# Higher anomaly score means more anomalous
calibration_scores = -model.decision_function(
    X_calibration
)

threshold = np.quantile(
    calibration_scores,
    CALIBRATION_QUANTILE
)

print("Calibration quantile:", CALIBRATION_QUANTILE)
print("Anomaly threshold:", threshold)

# --------------------------------------------------
# TEST
# --------------------------------------------------

print("\nEvaluating unknown attacks...")

start = time.perf_counter()

test_scores = -model.decision_function(X_test)

inference_time = time.perf_counter() - start

# 1 = detected anomaly, 0 = normal
y_pred = (
    test_scores > threshold
).astype(int)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0

roc_auc = roc_auc_score(
    y_test, test_scores
)

pr_auc = average_precision_score(
    y_test, test_scores
)

print("\n========== TRACK A RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"FPR:       {fpr:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nConfusion Matrix")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["BENIGN", "UNKNOWN_ATTACK"],
        zero_division=0
    )
)

print("\nPer unknown attack family")

results = []

for family in sorted(test.loc[
    test["label"] != "BENIGN", "label"
].unique()):

    mask = test["label"].to_numpy() == family

    family_recall = recall_score(
        y_test[mask],
        y_pred[mask],
        zero_division=0
    )

    detected = int(y_pred[mask].sum())
    total = int(mask.sum())

    results.append({
        "family": family,
        "total": total,
        "detected": detected,
        "recall": family_recall
    })

    print(
        f"{family}: {detected}/{total} "
        f"detected | Recall: {family_recall:.4f}"
    )

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

joblib.dump(
    model,
    OUTPUT / "track_a_isolation_forest.joblib"
)

pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "anomaly_score": test_scores
}).to_csv(
    OUTPUT / "track_a_predictions.csv",
    index=False
)

pd.DataFrame(results).to_csv(
    OUTPUT / "track_a_family_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "fpr": fpr,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "threshold": threshold,
    "training_time_seconds": training_time,
    "inference_time_seconds": inference_time
}])

summary.to_csv(
    OUTPUT / "track_a_summary.csv",
    index=False
)

print("\nResults saved in:", OUTPUT)
print("TRACK A COMPLETED")
