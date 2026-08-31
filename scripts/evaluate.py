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

DEVICE = torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

print("=" * 60)
print("EVALUATION DU MODELE")
print("=" * 60)

test_dataset = DeforestationDataset(TEST_DIR)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Nombre de données de test :", len(test_dataset))


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

model = model.to(DEVICE)

model.eval()

print("Modèle chargé :", MODEL_PATH)

print(
    "Epoch du modèle sauvegardé :",
    checkpoint["epoch"]
)


# ============================================================
# METRIQUES
# ============================================================

total_dice = 0.0
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0

num_batches = 0


# ============================================================
# EVALUATION
# ============================================================

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities > 0.5
        ).float()

        targets = masks.unsqueeze(1)


        # ----------------------------------------------------
        # Calcul des éléments
        # ----------------------------------------------------

        true_positive = (
            predictions * targets
        ).sum().item()

        false_positive = (
            predictions * (1 - targets)
        ).sum().item()

        false_negative = (
            (1 - predictions) * targets
        ).sum().item()


        # ----------------------------------------------------
        # Dice
        # ----------------------------------------------------

        dice = (
            2 * true_positive + 1
        ) / (
            2 * true_positive
            + false_positive
            + false_negative
            + 1
        )


        # ----------------------------------------------------
        # IoU
        # ----------------------------------------------------

        iou = (
            true_positive + 1
        ) / (
            true_positive
            + false_positive
            + false_negative
            + 1
        )


        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision = (
            true_positive + 1
        ) / (
            true_positive
            + false_positive
            + 1
        )


        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall = (
            true_positive + 1
        ) / (
            true_positive
            + false_negative
            + 1
        )


        total_dice += dice
        total_iou += iou
        total_precision += precision
        total_recall += recall

        num_batches += 1


# ============================================================
# RESULTATS
# ============================================================

dice = total_dice / num_batches
iou = total_iou / num_batches
precision = total_precision / num_batches
recall = total_recall / num_batches


print()
print("=" * 60)
print("RESULTATS SUR LE TEST")
print("=" * 60)

print(
    f"Dice      : {dice:.4f}"
)

print(
    f"IoU       : {iou:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print("=" * 60)
