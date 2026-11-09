
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "set_01"
OUTPUT = BASE / "processed_data"
OUTPUT.mkdir(exist_ok=True)

CHUNK_SIZE = 100_000
BENIGN_LIMIT = 100_000
ROLLING_WINDOW = 20

KNOWN_FILES = {
    "DoS": "DoS",
    "force-neutral": "FORCE_NEUTRAL",
    "rpm": "RPM",
    "standstill": "STANDSTILL",
    "targeted_payload_fuzzing": "FUZZING_TARGETED"
}

UNKNOWN_FILES = {
    "double": "DOUBLE",
    "fuzzing": "FUZZING",
    "interval": "INTERVAL",
    "speed": "SPEED",
    "systematic": "SYSTEMATIC",
    "triple": "TRIPLE"
}

FEATURE_COLUMNS = (
    ["timestamp", "can_id", "payload_length"]
    + [f"byte_{i}" for i in range(8)]
    + [
        "delta_t_ms",
        "same_id_delta_t_ms",
        "rolling_mean_delta_t_ms"
    ]
)


# --------------------------------------------------
# TEMPORAL STATE
# --------------------------------------------------

class TemporalState:

    def __init__(self):
        self.previous_timestamp = None
        self.last_timestamp_by_id = {}
        self.recent_deltas = deque(maxlen=ROLLING_WINDOW)

    def calculate(self, timestamps, can_ids):

        delta_values = []
        same_id_values = []
        rolling_values = []

        for timestamp, can_id in zip(timestamps, can_ids):

            if self.previous_timestamp is None:
                delta = 0.0
            else:
                delta = max(
                    0.0,
                    (timestamp - self.previous_timestamp) * 1000
                )

            self.previous_timestamp = timestamp

            if can_id in self.last_timestamp_by_id:
                same_delta = max(
                    0.0,
                    (timestamp - self.last_timestamp_by_id[can_id]) * 1000
                )
            else:
                same_delta = 0.0

            self.last_timestamp_by_id[can_id] = timestamp

            self.recent_deltas.append(delta)
            rolling_mean = float(np.mean(self.recent_deltas))

            delta_values.append(delta)
            same_id_values.append(same_delta)
            rolling_values.append(rolling_mean)

        return delta_values, same_id_values, rolling_values


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(df, temporal_state):

    result = pd.DataFrame(index=df.index)

    result["timestamp"] = pd.to_numeric(
        df["timestamp"], errors="coerce"
    )

    result["can_id"] = df["arbitration_id"].apply(
        lambda x: int(str(x), 16) if pd.notna(x) else np.nan
    )

    payloads = (
        df["data_field"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["payload_length"] = payloads.str.len() // 2

    for i in range(8):
        values = payloads.str[i * 2:(i + 1) * 2]

        result[f"byte_{i}"] = values.apply(
            lambda x: int(x, 16) if len(x) == 2 else 0
        )

    result["attack"] = pd.to_numeric(
        df["attack"], errors="coerce"
    )

    valid = result.dropna(
        subset=["timestamp", "can_id", "attack"]
    ).copy()

    (
        valid["delta_t_ms"],
        valid["same_id_delta_t_ms"],
        valid["rolling_mean_delta_t_ms"]
    ) = temporal_state.calculate(
        valid["timestamp"].to_numpy(),
        valid["can_id"].to_numpy()
    )

    return valid


# --------------------------------------------------
# PROCESS ONE ORIGINAL CAPTURE FILE
# --------------------------------------------------

def process_file(path, split, family):

    print(f"\nProcessing: {path.name}")

    rng = np.random.default_rng(42)

    # A fresh temporal state for every original capture
    temporal_state = TemporalState()

    benign_parts = []
    attack_parts = []

    benign_seen = 0
    attack_seen = 0
    negative_timestamp_differences = 0
    previous_timestamp = None

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        dtype=str
    ):

        features = extract_features(
            chunk,
            temporal_state
        )

        # Count negative timestamp transitions accurately,
        # including transitions between consecutive chunks.
        timestamps = features["timestamp"].to_numpy()

        if len(timestamps):

            if previous_timestamp is not None:
                if timestamps[0] < previous_timestamp:
                    negative_timestamp_differences += 1

            negative_timestamp_differences += int(
                np.sum(np.diff(timestamps) < 0)
            )

            previous_timestamp = timestamps[-1]

        # Retain all attack records
        attacks = features[
            features["attack"] == 1
        ]

        if len(attacks):

            attacks = attacks.copy()
            attacks["label"] = family

            attack_parts.append(attacks)
            attack_seen += len(attacks)

        # IMPORTANT:
        # Sample benign traffic from EVERY capture,
        # including captures that also contain attacks.
        benign = features[
            features["attack"] == 0
        ]

        if len(benign):

            benign_seen += len(benign)

            sample = benign.sample(
                frac=0.10,
                random_state=int(
                    rng.integers(0, 2**31 - 1)
                )
            )

            sample = sample.copy()
            sample["label"] = "BENIGN"

            benign_parts.append(sample)

    parts = benign_parts + attack_parts

    if not parts:
        print("WARNING: No records found.")
        return

    output = pd.concat(
        parts,
        ignore_index=True
    )

    # Apply benign-record limit per original capture
    if benign_parts:

        benign_output = output[
            output["label"] == "BENIGN"
        ]

        if len(benign_output) > BENIGN_LIMIT:

            benign_output = benign_output.sample(
                n=BENIGN_LIMIT,
                random_state=42
            )

        attack_output = output[
            output["label"] != "BENIGN"
        ]

        output = pd.concat(
            [benign_output, attack_output],
            ignore_index=True
        )

    output["split"] = split

    output = output[
        FEATURE_COLUMNS + ["label", "split"]
    ]

    output_path = OUTPUT / f"{split}_{path.stem}.csv"

    output.to_csv(
        output_path,
        index=False
    )

    print(f"Benign available: {benign_seen:,}")
    print(f"Attack retained: {attack_seen:,}")
    print(f"Negative timestamp transitions: {negative_timestamp_differences:,}")
    print(f"Records saved: {len(output):,}")
    print(f"Saved: {output_path}")


# --------------------------------------------------
# TRAINING DATA
# --------------------------------------------------

train_folder = ROOT / "train_02_with_attacks"

for filename in ["attack-free-1.csv", "accessory-1.csv"]:

    process_file(
        train_folder / filename,
        "train",
        "BENIGN"
    )

for prefix, family in KNOWN_FILES.items():

    process_file(
        train_folder / f"{prefix}-1.csv",
        "train",
        family
    )


# --------------------------------------------------
# CALIBRATION DATA
# --------------------------------------------------

for filename in ["attack-free-2.csv", "accessory-2.csv"]:

    process_file(
        train_folder / filename,
        "calibration",
        "BENIGN"
    )

for prefix, family in KNOWN_FILES.items():

    process_file(
        train_folder / f"{prefix}-2.csv",
        "calibration",
        family
    )


# --------------------------------------------------
# UNKNOWN-ATTACK TEST DATA
# --------------------------------------------------

test_folder = ROOT / "test_03_known_vehicle_unknown_attack"

for prefix, family in UNKNOWN_FILES.items():

    for suffix in ["3", "4"]:

        process_file(
            test_folder / f"{prefix}-{suffix}.csv",
            "unknown_test",
            family
        )


# --------------------------------------------------
# COMBINE AND AUDIT
# --------------------------------------------------

print("\nCombining processed datasets...")

for split in ["train", "calibration", "unknown_test"]:

    files = sorted(
        OUTPUT.glob(f"{split}_*.csv")
    )

    if not files:
        print(f"No files found for {split}")
        continue

    combined = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    final_path = OUTPUT / f"{split}.csv"

    combined.to_csv(
        final_path,
        index=False
    )

    print(f"\n{split.upper()}")
    print(f"Total records: {len(combined):,}")
    print(combined["label"].value_counts().to_string())
    print(f"Saved: {final_path}")

print("\nPREPROCESSING COMPLETED")
