
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
THRESHOLD = 0.0243811395

# Load existing unknown-test predictions and scores
df = pd.read_csv(RESULTS / "track_b_embedding_predictions.csv")

# Ground truth: BENIGN = 0, every attack family = 1
y_true = (df["actual"] != "BENIGN").astype(int)

# Higher anomaly score means more anomalous
y_pred = (df["anomaly_score"] > THRESHOLD).astype(int)

# Calculate metrics
tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
fpr = fp / (fp + tn) if (fp + tn) else 0

roc_auc = roc_auc_score(y_true, df["anomaly_score"])
pr_auc = average_precision_score(y_true, df["anomaly_score"])

print("\n=== CALIBRATION-SELECTED THRESHOLD: TEST RESULTS ===")
print(f"Fixed threshold: {THRESHOLD:.10f}")
print(f"Test records: {len(df)}")
print(f"Accuracy:  {accuracy:.6f} ({accuracy:.2%})")
print(f"Precision: {precision:.6f} ({precision:.2%})")
print(f"Recall:    {recall:.6f} ({recall:.2%})")
print(f"F1-score:  {f1:.6f} ({f1:.2%})")
print(f"FPR:       {fpr:.6f} ({fpr:.2%})")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")

print(f"\nConfusion matrix:")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\n--- Per unknown attack family ---")
attack_rows = []

for family in sorted(df.loc[df["actual"] != "BENIGN", "actual"].unique()):
    subset = df[df["actual"] == family]
    detected = int((subset["anomaly_score"] > THRESHOLD).sum())
    total = len(subset)
    family_recall = detected / total if total else 0

    print(f"{family}: {detected}/{total} ({family_recall:.2%})")

    attack_rows.append({
        "family": family,
        "detected": detected,
        "total": total,
        "recall": family_recall
    })

# Save results
summary = pd.DataFrame([{
    "threshold": THRESHOLD,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "fpr": fpr,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp
}])

summary.to_csv(RESULTS / "track_b_embedding_calibrated_summary.csv", index=False)

pd.DataFrame(attack_rows).to_csv(
    RESULTS / "track_b_embedding_calibrated_family_results.csv",
    index=False
)

print("\nSaved calibrated test results under results/:")
print("track_b_embedding_calibrated_summary.csv")
print("track_b_embedding_calibrated_family_results.csv")
