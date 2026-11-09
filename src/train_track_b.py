
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "processed_data"
OUTPUT = BASE / "results"
OUTPUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
EPOCHS = 25
BATCH_SIZE = 4096
LEARNING_RATE = 0.001
TEMPERATURE = 1.0
CALIBRATION_QUANTILE = 0.99

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cpu")

FEATURES = (
    ["can_id", "payload_length"]
    + [f"byte_{i}" for i in range(8)]
    + [
        "delta_t_ms",
        "same_id_delta_t_ms",
        "rolling_mean_delta_t_ms"
    ]
)

CLASSES = [
    "BENIGN",
    "DoS",
    "FORCE_NEUTRAL",
    "FUZZING_TARGETED",
    "RPM",
    "STANDSTILL"
]

CLASS_TO_ID = {
    name: i for i, name in enumerate(CLASSES)
}

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading datasets...")

train = pd.read_csv(DATA / "train.csv")
calibration = pd.read_csv(DATA / "calibration.csv")
test = pd.read_csv(DATA / "unknown_test.csv")

X_train = train[FEATURES].to_numpy(dtype=np.float32)
y_train = train["label"].map(CLASS_TO_ID).to_numpy()

X_cal = calibration[FEATURES].to_numpy(dtype=np.float32)
y_cal = calibration["label"].map(CLASS_TO_ID).to_numpy()

X_test = test[FEATURES].to_numpy(dtype=np.float32)

y_test = (
    test["label"] != "BENIGN"
).astype(int).to_numpy()

if np.isnan(y_train).any() or np.isnan(y_cal).any():
    raise ValueError("Unexpected class labels found.")

y_train = y_train.astype(np.int64)
y_cal = y_cal.astype(np.int64)

print("\nTraining class distribution:")
print(train["label"].value_counts())

print("\nCalibration class distribution:")
print(calibration["label"].value_counts())

print("\nUnknown test records:", len(test))
print("Unknown attacks:", int(y_test.sum()))

# --------------------------------------------------
# SCALE FEATURES
# Fit scaler ONLY on training data
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_cal = scaler.transform(X_cal).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# --------------------------------------------------
# CREATE PYTORCH DATASET
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# --------------------------------------------------
# NEURAL NETWORK
# --------------------------------------------------

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
    input_size=len(FEATURES),
    number_of_classes=len(CLASSES)
).to(DEVICE)

# --------------------------------------------------
# CLASS WEIGHTS
# Square-root inverse-frequency weighting
# --------------------------------------------------

class_counts = np.bincount(
    y_train,
    minlength=len(CLASSES)
)

class_weights = np.sqrt(
    len(y_train) / (len(CLASSES) * class_counts)
)

class_weights = class_weights / class_weights.mean()

weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass weights:")

for name, weight in zip(CLASSES, class_weights):
    print(f"{name}: {weight:.4f}")

criterion = nn.CrossEntropyLoss(
    weight=weights_tensor
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nTraining neural network...")

start = time.perf_counter()

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_records = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_y)

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == batch_y
        ).sum().item()

        total_records += len(batch_y)

    average_loss = total_loss / total_records
    train_accuracy = total_correct / total_records

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {average_loss:.4f} "
        f"| Training accuracy: {train_accuracy:.4f}"
    )

training_time = time.perf_counter() - start

print(f"\nTraining time: {training_time:.2f} seconds")

# --------------------------------------------------
# ENERGY SCORE
# E(x) = -T * logsumexp(logits / T)
# Higher energy indicates a less familiar input.
# --------------------------------------------------

def calculate_energy(model, features):

    model.eval()

    all_logits = []

    with torch.no_grad():

        for start_index in range(0, len(features), 8192):

            batch = torch.tensor(
                features[start_index:start_index + 8192],
                dtype=torch.float32
            ).to(DEVICE)

            logits = model(batch)

            all_logits.append(
                logits.cpu().numpy()
            )

    logits = np.concatenate(all_logits, axis=0)

    energy = -TEMPERATURE * np.logaddexp.reduce(
        logits / TEMPERATURE,
        axis=1
    )

    return energy, logits


# --------------------------------------------------
# CALIBRATION
# --------------------------------------------------

print("\nCalculating calibration energy...")

cal_energy, cal_logits = calculate_energy(
    model,
    X_cal
)

threshold = np.quantile(
    cal_energy,
    CALIBRATION_QUANTILE
)

print("Calibration quantile:", CALIBRATION_QUANTILE)
print("Energy threshold:", threshold)

# Also report calibration class distribution above threshold
cal_above = cal_energy > threshold

print("\nCalibration records above threshold:")

for class_id, class_name in enumerate(CLASSES):

    mask = y_cal == class_id

    if mask.sum() == 0:
        continue

    count = int((cal_above & mask).sum())

    print(
        f"{class_name}: {count}/{int(mask.sum())} "
        f"({count / mask.sum():.4%})"
    )

# --------------------------------------------------
# UNKNOWN-ATTACK TEST
# --------------------------------------------------

print("\nEvaluating unknown attacks...")

start = time.perf_counter()

test_energy, test_logits = calculate_energy(
    model,
    X_test
)

inference_time = time.perf_counter() - start

y_pred = (
    test_energy > threshold
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
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0

roc_auc = roc_auc_score(
    y_test,
    test_energy
)

pr_auc = average_precision_score(
    y_test,
    test_energy
)

print("\n========== TRACK B RESULTS ==========")

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

# --------------------------------------------------
# PER UNKNOWN ATTACK FAMILY
# --------------------------------------------------

print("\nPer unknown attack family:")

family_results = []

test_labels = test["label"].to_numpy()

for family in sorted(
    test.loc[
        test["label"] != "BENIGN",
        "label"
    ].unique()
):

    mask = test_labels == family

    detected = int(y_pred[mask].sum())
    total = int(mask.sum())

    family_recall = detected / total

    family_results.append({
        "family": family,
        "total": total,
        "detected": detected,
        "recall": family_recall
    })

    print(
        f"{family}: {detected}/{total} "
        f"| Recall: {family_recall:.4f}"
    )

# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "features": FEATURES,
        "temperature": TEMPERATURE,
        "threshold": float(threshold)
    },
    OUTPUT / "track_b_energy_model.pth"
)

joblib.dump(
    scaler,
    OUTPUT / "track_b_scaler.joblib"
)

pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred,
    "energy": test_energy
}).to_csv(
    OUTPUT / "track_b_predictions.csv",
    index=False
)

pd.DataFrame(
    family_results
).to_csv(
    OUTPUT / "track_b_family_results.csv",
    index=False
)

pd.DataFrame([{
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
}]).to_csv(
    OUTPUT / "track_b_summary.csv",
    index=False
)

print("\nResults saved in:", OUTPUT)
print("TRACK B COMPLETED")
