
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "processed_data"

df = pd.read_csv(DATA / "unknown_test.csv")

families = [
    "BENIGN", "DOUBLE", "FUZZING",
    "INTERVAL", "SPEED", "SYSTEMATIC", "TRIPLE"
]

features = [
    "delta_t_ms",
    "same_id_delta_t_ms",
    "rolling_mean_delta_t_ms",
    "payload_length"
]

print("\n=== TEMPORAL FEATURE ANALYSIS ===")

summary = []

for family in families:
    subset = df[df["label"] == family]

    if subset.empty:
        continue

    print(f"\n{family}: {len(subset)} records")

    row = {"family": family, "count": len(subset)}

    for feature in features:
        if feature not in subset.columns:
            print(f"{feature}: NOT FOUND")
            continue

        values = subset[feature].dropna()

        row[f"{feature}_median"] = values.median()
        row[f"{feature}_mean"] = values.mean()
        row[f"{feature}_std"] = values.std()

        print(
            f"{feature}: "
            f"median={values.median():.4f}, "
            f"mean={values.mean():.4f}, "
            f"std={values.std():.4f}"
        )

    summary.append(row)

RESULTS.mkdir(parents=True, exist_ok=True)

pd.DataFrame(summary).to_csv(
    str(RESULTS / "attack_temporal_feature_analysis.csv"),
    index=False
)

print("\nSaved: results/attack_temporal_feature_analysis.csv")
