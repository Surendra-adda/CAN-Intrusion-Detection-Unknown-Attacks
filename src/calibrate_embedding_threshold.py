
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

# Load benign calibration anomaly scores
df = pd.read_csv(RESULTS / "track_b_embedding_calibration_scores.csv")

scores = df["anomaly_score"].dropna().to_numpy()

# Predefined operating point: 2.5% calibration exceedance rate
threshold = np.percentile(scores, 97.5)

calibration_fpr = np.mean(scores > threshold)

print("\n=== CALIBRATION-ONLY THRESHOLD ===")
print(f"Calibration records: {len(scores)}")
print(f"Selected percentile: 97.5")
print(f"Threshold: {threshold:.10f}")
print(f"Calibration FPR: {calibration_fpr:.4%}")

# Save threshold separately
output = pd.DataFrame([{
    "percentile": 97.5,
    "threshold": threshold,
    "calibration_fpr": calibration_fpr,
    "calibration_records": len(scores)
}])

output.to_csv(
    RESULTS / "calibration_threshold_97_5.csv",
    index=False
)

print("\nSaved: results/calibration_threshold_97_5.csv")
