import torch
from torch.utils.data import DataLoader

from scripts.dataset import DeforestationDataset
from scripts.model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "data/ml_dataset/test"
MODEL_PATH = "models/best_model.pth"

BATCH_SIZE = 2

THRESHOLD = 0.70

DEVICE = torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

print("=" * 60)
print("EVALUATION FINALE AVEC SEUIL OPTIMISE")
print("=" * 60)

dataset = DeforestationDataset(TEST_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Nombre de données de test :", len(dataset))


# ============================================================
# MODELE
# ============================================================

model = UNet(
    in_channels=8,
    out_channels=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Modèle :", MODEL_PATH)
print("Epoch :", checkpoint["epoch"])
print("Seuil :", THRESHOLD)


# ============================================================
# COMPTEURS GLOBAUX
# ============================================================

true_positive = 0
false_positive = 0
false_negative = 0


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        targets = masks.unsqueeze(1)


        true_positive += (
            predictions * targets
        ).sum().item()

        false_positive += (
            predictions * (1 - targets)
        ).sum().item()

        false_negative += (
            (1 - predictions) * targets
        ).sum().item()


# ============================================================
# METRIQUES
# ============================================================

dice = (
    2 * true_positive
) / (
    2 * true_positive
    + false_positive
    + false_negative
    + 1e-8
)

iou = (
    true_positive
) / (
    true_positive
    + false_positive
    + false_negative
    + 1e-8
)

precision = (
    true_positive
) / (
    true_positive
    + false_positive
    + 1e-8
)

recall = (
    true_positive
) / (
    true_positive
    + false_negative
    + 1e-8
)


# ============================================================
# RESULTATS
# ============================================================

print()
print("=" * 60)
print("RESULTATS FINAUX SUR LE TEST")
print("=" * 60)

print(f"Dice      : {dice:.4f}")
print(f"IoU       : {iou:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")

print()
print("Pixels TP :", int(true_positive))
print("Pixels FP :", int(false_positive))
print("Pixels FN :", int(false_negative))

print("=" * 60)

