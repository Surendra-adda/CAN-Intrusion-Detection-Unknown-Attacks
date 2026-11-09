
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
DATA = BASE / "processed_data"

cal = pd.read_csv(RESULTS / "track_b_embedding_predictions.csv")
test = cal

# Recalculate calibration scores using the existing detector.
# This script requires the calibration scores from the
# embedding experiment to be saved first.

cal_scores_path = RESULTS / "track_b_embedding_calibration_scores.csv"

if not cal_scores_path.exists():
    raise FileNotFoundError(
        "Calibration scores are missing. "
        "We must save them from the original embedding experiment."
    )

cal_scores = pd.read_csv(cal_scores_path)["anomaly_score"].to_numpy()

test_scores = pd.read_csv(
    RESULTS / "track_b_embedding_predictions.csv"
)["anomaly_score"].to_numpy()

test_df = pd.read_csv(DATA / "unknown_test.csv")

y_true = (test_df["label"] != "BENIGN").astype(int).to_numpy()

print("\n========== EMBEDDING THRESHOLD ANALYSIS ==========")

for percentile in [90, 95, 97.5, 99]:

    threshold = np.percentile(cal_scores, percentile)
    y_pred = (test_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn)

    print(f"\nCalibration percentile: {percentile}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision:.4%}")
    print(f"Recall: {recall:.4%}")
    print(f"F1: {f1:.4%}")
    print(f"FPR: {fpr:.4%}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    print("Attack family recall:")

    for family in sorted(test_df["label"].unique()):
        if family == "BENIGN":
            continue

        mask = (test_df["label"] == family).to_numpy()
        detected = int(y_pred[mask].sum())
        total = int(mask.sum())

        print(
            f"  {family}: {detected}/{total} "
            f"({detected / total:.2%})"
        )

print("\nAnalysis completed.")
