"""
FedGAN-ID-inspired maximum-training-energy threshold experiment.

Run from the project root containing processed_data/ and results/.
This reuses the saved Track B model and scaler; it does not retrain.
It does not use unknown-test labels to select the threshold.
"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "processed_data"
RESULTS = BASE / "results"
MODEL_PATH = RESULTS / "track_b_energy_model.pth"
SCALER_PATH = RESULTS / "track_b_scaler.joblib"
DEVICE = torch.device("cpu")
TEMPERATURE = 1.0
BATCH_SIZE = 8192


class CANClassifier(nn.Module):
    def __init__(self, input_size, number_of_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, number_of_classes)
        )

    def forward(self, x):
        return self.network(x)


def calculate_energy(model, features):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), BATCH_SIZE):
            batch = torch.tensor(
                features[start:start + BATCH_SIZE],
                dtype=torch.float32, device=DEVICE
            )
            logits = model(batch)
            energy = -TEMPERATURE * torch.logsumexp(
                logits / TEMPERATURE, dim=1
            )
            chunks.append(energy.cpu().numpy())
    return np.concatenate(chunks)


def main():
    print("\n=== FedGAN-ID-inspired maximum training-energy threshold ===")
    train_path = DATA / "train.csv"
    test_path = DATA / "unknown_test.csv"
    for p in [train_path, test_path, MODEL_PATH, SCALER_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    features = checkpoint["features"]
    classes = checkpoint["classes"]
    scaler = joblib.load(SCALER_PATH)

    # Paper describes setting tau to the maximum training energy and says
    # vehicles upload logits for malicious training messages. We therefore
    # use all non-BENIGN known attack rows in our training split.
    attack_train = train.loc[train["label"] != "BENIGN"].copy()
    if attack_train.empty:
        raise ValueError("No non-BENIGN training rows found.")
    unexpected = sorted(set(attack_train["label"]) - set(classes))
    if unexpected:
        raise ValueError(f"Training labels absent from model classes: {unexpected}")

    model = CANClassifier(len(features), len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_attack = scaler.transform(
        attack_train[features].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    x_test = scaler.transform(
        test[features].to_numpy(dtype=np.float32)
    ).astype(np.float32)

    print("Known malicious training records:", len(attack_train))
    print("Unknown test records:", len(test))
    print("Calculating training attack energies...")
    train_energy = calculate_energy(model, x_attack)
    threshold = float(np.max(train_energy))
    print("Maximum training attack energy threshold:", threshold)

    print("Scoring unknown-test records...")
    started = time.perf_counter()
    test_energy = calculate_energy(model, x_test)
    inference_seconds = time.perf_counter() - started

    y_true = (test["label"] != "BENIGN").astype(int).to_numpy()
    y_pred = (test_energy > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "method": "FedGAN-ID-inspired energy (max known-attack train energy)",
        "threshold_policy": "maximum energy among non-BENIGN training rows",
        "threshold": threshold,
        "training_attack_records": len(attack_train),
        "test_records": len(test),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "roc_auc": roc_auc_score(y_true, test_energy),
        "pr_auc": average_precision_score(y_true, test_energy),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "inference_time_seconds": inference_seconds,
    }

    print("\n--- Results ---")
    for key in ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]:
        print(f"{key}: {metrics[key]:.6f} ({metrics[key]:.2%})" if key not in ["roc_auc", "pr_auc"]
              else f"{key}: {metrics[key]:.6f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Inference time: {inference_seconds:.2f} seconds")

    family_rows = []
    print("\n--- Per unknown attack family ---")
    for family in sorted(test.loc[test["label"] != "BENIGN", "label"].unique()):
        mask = test["label"].to_numpy() == family
        total = int(mask.sum())
        detected = int(y_pred[mask].sum())
        recall = detected / total if total else 0.0
        family_rows.append({
            "family": family, "total": total,
            "detected": detected, "recall": recall
        })
        print(f"{family}: {detected}/{total} ({recall:.2%})")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        RESULTS / "fedgan_max_train_energy_summary.csv", index=False
    )
    pd.DataFrame(family_rows).to_csv(
        RESULTS / "fedgan_max_train_energy_family_results.csv", index=False
    )
    pd.DataFrame({
        "actual_label": test["label"].to_numpy(),
        "actual_binary": y_true,
        "predicted_binary": y_pred,
        "energy": test_energy
    }).to_csv(
        RESULTS / "fedgan_max_train_energy_predictions.csv", index=False
    )

    # Bring in existing summaries when present; do not recompute or alter them.
    comparison = [metrics]
    existing = [
        (RESULTS / "track_b_summary.csv", "Our energy OOD baseline",
         "99th percentile of full known-class calibration energy"),
        (RESULTS / "track_b_embedding_summary.csv", "Representation + Isolation Forest",
         "99th percentile of benign calibration anomaly scores"),
    ]
    for path, method, policy in existing:
        if path.exists():
            old = pd.read_csv(path)
            if not old.empty:
                row = old.iloc[0].to_dict()
                row["method"] = method
                row["threshold_policy"] = policy
                comparison.append(row)
    pd.DataFrame(comparison).to_csv(
        RESULTS / "energy_method_comparison.csv", index=False
    )

    print("\nSaved new results under results/:")
    print("fedgan_max_train_energy_summary.csv")
    print("fedgan_max_train_energy_family_results.csv")
    print("fedgan_max_train_energy_predictions.csv")
    print("energy_method_comparison.csv")
    print("\nThis is a threshold-policy adaptation, not a full FedGAN-ID reproduction.")
    print("No GAN, federated learning, or differential privacy is implemented.")


if __name__ == "__main__":
    main()
