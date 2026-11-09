
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# ==================================================
# PATHS
# ==================================================

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "processed_data"
RESULTS = BASE / "results"

MODEL_PATH = RESULTS / "track_b_energy_model.pth"
SCALER_PATH = RESULTS / "track_b_scaler.joblib"

DEVICE = torch.device("cpu")

# ==================================================
# LOAD MODEL CHECKPOINT
# ==================================================

print("\n========== TRACK B HIDDEN REPRESENTATION TEST ==========")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

FEATURES = checkpoint["features"]
CLASSES = checkpoint["classes"]

print("Features:", FEATURES)
print("Classes:", CLASSES)

# ==================================================
# REBUILD EXACT MODEL ARCHITECTURE
# ==================================================

class CANClassifier(nn.Module):

    def __init__(self, input_size, number_of_classes):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, number_of_classes)
        )

    def forward(self, x):
        return self.network(x)


model = CANClassifier(
    len(FEATURES),
    len(CLASSES)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# ==================================================
# EXTRACT 64-DIMENSIONAL REPRESENTATIONS
# ==================================================

hidden_outputs = []

def capture_hidden_layer(module, inputs, output):
    hidden_outputs.append(output.detach().cpu().numpy())


hook = model.network[4].register_forward_hook(capture_hidden_layer)


def extract_embeddings(df, scaler, batch_size=8192):

    X = df[FEATURES].to_numpy(dtype=np.float32)
    X = scaler.transform(X).astype(np.float32)

    all_embeddings = []

    with torch.no_grad():

        for start in range(0, len(X), batch_size):

            batch = torch.tensor(
                X[start:start + batch_size],
                dtype=torch.float32
            )

            hidden_outputs.clear()

            model(batch)

            embedding = hidden_outputs[-1]
            all_embeddings.append(embedding)

    return np.vstack(all_embeddings)


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading datasets...")

train = pd.read_csv(DATA / "train.csv")
cal = pd.read_csv(DATA / "calibration.csv")
test = pd.read_csv(DATA / "unknown_test.csv")

scaler = joblib.load(SCALER_PATH)

train_benign = train[train["label"] == "BENIGN"].copy()
cal_benign = cal[cal["label"] == "BENIGN"].copy()

print("Training benign:", len(train_benign))
print("Calibration benign:", len(cal_benign))
print("Unknown test:", len(test))

# ==================================================
# EXTRACT REPRESENTATIONS
# ==================================================

print("\nExtracting training representations...")
Z_train = extract_embeddings(train_benign, scaler)

print("\nExtracting calibration representations...")
Z_cal = extract_embeddings(cal_benign, scaler)

print("\nExtracting unknown test representations...")
Z_test = extract_embeddings(test, scaler)

hook.remove()

print("\nRepresentation shapes:")
print("Train:", Z_train.shape)
print("Calibration:", Z_cal.shape)
print("Test:", Z_test.shape)

# ==================================================
# TRAIN BENIGN-ONLY ANOMALY DETECTOR
# ==================================================

print("\nTraining Isolation Forest on hidden representations...")

detector = IsolationForest(
    n_estimators=300,
    max_samples=2048,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

detector.fit(Z_train)

# Higher score means more anomalous
cal_scores = -detector.decision_function(Z_cal)
test_scores = -detector.decision_function(Z_test)
pd.DataFrame({
    "anomaly_score": cal_scores
}).to_csv(
    RESULTS / "track_b_embedding_calibration_scores.csv",
    index=False
)

print("Calibration scores saved successfully.")
# Threshold uses benign calibration data only
threshold = np.percentile(cal_scores, 99)

print("\nCalibration threshold:", threshold)

# ==================================================
# EVALUATION
# ==================================================

y_true = (test["label"] != "BENIGN").astype(int).to_numpy()
y_pred = (test_scores > threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

fpr = fp / (fp + tn) if (fp + tn) else 0

roc_auc = roc_auc_score(y_true, test_scores)
pr_auc = average_precision_score(y_true, test_scores)

print("\n========== EMBEDDING DETECTOR RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"FPR:       {fpr:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nConfusion matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ==================================================
# ATTACK FAMILY EVALUATION
# ==================================================

print("\n========== ATTACK FAMILY RESULTS ==========")

family_results = []

for family in sorted(test["label"].unique()):

    if family == "BENIGN":
        continue

    mask = (test["label"] == family).to_numpy()

    detected = int(y_pred[mask].sum())
    total = int(mask.sum())

    family_recall = detected / total if total else 0

    print(
        f"{family}: {detected}/{total} "
        f"({family_recall:.4%})"
    )

    family_results.append({
        "family": family,
        "detected": detected,
        "total": total,
        "recall": family_recall
    })

# ==================================================
# SAVE RESULTS
# ==================================================

RESULTS.mkdir(exist_ok=True)

pd.DataFrame({
    "actual": test["label"],
    "anomaly_score": test_scores,
    "predicted": y_pred
}).to_csv(
    RESULTS / "track_b_embedding_predictions.csv",
    index=False
)

pd.DataFrame(family_results).to_csv(
    RESULTS / "track_b_embedding_family_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "threshold": threshold,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "fpr": fpr,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc
}])

summary.to_csv(
    RESULTS / "track_b_embedding_summary.csv",
    index=False
)

print("\nResults saved successfully.")
print("========== EXPERIMENT COMPLETED ==========")
