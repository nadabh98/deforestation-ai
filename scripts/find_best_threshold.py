import torch
from torch.utils.data import DataLoader

from scripts.dataset import DeforestationDataset
from scripts.model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_DIR = "data/ml_dataset/validation"
MODEL_PATH = "models/best_model.pth"

BATCH_SIZE = 2

DEVICE = torch.device("cpu")

# Seuils que nous allons tester
THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


# ============================================================
# DATASET
# ============================================================

print("=" * 60)
print("RECHERCHE DU MEILLEUR SEUIL")
print("=" * 60)

dataset = DeforestationDataset(VALIDATION_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Nombre de données :", len(dataset))


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


# ============================================================
# STOCKAGE DES PROBABILITES
# ============================================================

all_probabilities = []
all_targets = []


print()
print("Calcul des probabilités...")


with torch.no_grad():

    for images, masks in loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        all_probabilities.append(
            probabilities.cpu()
        )

        all_targets.append(
            masks.unsqueeze(1).cpu()
        )


probabilities = torch.cat(
    all_probabilities,
    dim=0
)

targets = torch.cat(
    all_targets,
    dim=0
)


# ============================================================
# FONCTION METRIQUES
# ============================================================

def calculate_metrics(
    probabilities,
    targets,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).float()

    true_positive = (
        predictions * targets
    ).sum().item()

    false_positive = (
        predictions * (1 - targets)
    ).sum().item()

    false_negative = (
        (1 - predictions) * targets
    ).sum().item()


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


    return dice, iou, precision, recall


# ============================================================
# TEST DES SEUILS
# ============================================================

print()
print("=" * 60)
print("RESULTATS PAR SEUIL")
print("=" * 60)

best_threshold = None
best_dice = -1

results = []


for threshold in THRESHOLDS:

    dice, iou, precision, recall = calculate_metrics(
        probabilities,
        targets,
        threshold
    )

    results.append(
        (
            threshold,
            dice,
            iou,
            precision,
            recall
        )
    )

    print(
        f"Seuil {threshold:.2f} | "
        f"Dice : {dice:.4f} | "
        f"IoU : {iou:.4f} | "
        f"Precision : {precision:.4f} | "
        f"Recall : {recall:.4f}"
    )


    if dice > best_dice:

        best_dice = dice

        best_threshold = threshold


# ============================================================
# MEILLEUR SEUIL
# ============================================================

print()
print("=" * 60)
print("MEILLEUR SEUIL")
print("=" * 60)

print(
    f"Seuil optimal : {best_threshold:.2f}"
)

print(
    f"Dice validation : {best_dice:.4f}"
)

print("=" * 60)
