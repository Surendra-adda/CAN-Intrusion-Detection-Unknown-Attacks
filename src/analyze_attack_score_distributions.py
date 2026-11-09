
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

df = pd.read_csv(
    RESULTS / "track_b_embedding_predictions.csv"
)

print("\n=== ANOMALY SCORE DISTRIBUTION ANALYSIS ===")

families = ["BENIGN", "DOUBLE", "FUZZING",
            "INTERVAL", "SPEED", "SYSTEMATIC", "TRIPLE"]

summary = []

for family in families:
    scores = df.loc[
        df["actual"] == family, "anomaly_score"
    ].dropna()

    if len(scores) == 0:
        continue

    row = {
        "family": family,
        "count": len(scores),
        "mean": scores.mean(),
        "median": scores.median(),
        "std": scores.std(),
        "min": scores.min(),
        "p25": scores.quantile(0.25),
        "p75": scores.quantile(0.75),
        "p90": scores.quantile(0.90),
        "p95": scores.quantile(0.95),
        "max": scores.max(),
        "detected_at_threshold": (scores > 0.0243811395).mean()
    }

    summary.append(row)

    print(f"\n{family}")
    print(f"Records: {row['count']}")
    print(f"Mean: {row['mean']:.6f}")
    print(f"Median: {row['median']:.6f}")
    print(f"Standard deviation: {row['std']:.6f}")
    print(f"25th percentile: {row['p25']:.6f}")
    print(f"75th percentile: {row['p75']:.6f}")
    print(f"90th percentile: {row['p90']:.6f}")
    print(f"95th percentile: {row['p95']:.6f}")
    print(f"Detection rate: {row['detected_at_threshold']:.2%}")

output = pd.DataFrame(summary)

output.to_csv(
    RESULTS / "attack_score_distribution_analysis.csv",
    index=False
)

print("\nSaved: results/attack_score_distribution_analysis.csv")
