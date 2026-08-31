import os

import torch
from torch.utils.data import DataLoader

from scripts.dataset import DeforestationDataset
from scripts.model import UNet
from scripts.losses import CombinedLoss


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "data/ml_dataset/train"
VAL_DIR = "data/ml_dataset/validation"

BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 1e-4

MODEL_DIR = "models"

DEVICE = torch.device("cpu")


# ============================================================
# CREATION DU DOSSIER MODELES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DATASETS
# ============================================================

print("=" * 60)
print("CHARGEMENT DES DATASETS")
print("=" * 60)

train_dataset = DeforestationDataset(TRAIN_DIR)

val_dataset = DeforestationDataset(VAL_DIR)

print("Train :", len(train_dataset))
print("Validation :", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODELE
# ============================================================

print()
print("=" * 60)
print("CREATION DU MODELE")
print("=" * 60)

model = UNet(
    in_channels=8,
    out_channels=1
)

model = model.to(DEVICE)

print(
    "Nombre de paramètres :",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# LOSS
# ============================================================

loss_fn = CombinedLoss()


# ============================================================
# OPTIMISEUR
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# METRIQUE DICE
# ============================================================

def dice_score(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).float()

    targets = targets.unsqueeze(1)

    intersection = (
        predictions * targets
    ).sum()

    dice = (
        2 * intersection + 1
    ) / (
        predictions.sum()
        + targets.sum()
        + 1
    )

    return dice.item()


# ============================================================
# METRIQUE IoU
# ============================================================

def iou_score(logits, targets):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > 0.5
    ).float()

    targets = targets.unsqueeze(1)

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum()

    iou = (
        intersection + 1
    ) / (
        union + 1
    )

    return iou.item()


# ============================================================
# ENTRAINEMENT
# ============================================================

best_val_loss = float("inf")


print()
print("=" * 60)
print("DEBUT DE L'ENTRAINEMENT")
print("=" * 60)

print("Device :", DEVICE)
print("Epochs :", EPOCHS)
print("Batch size :", BATCH_SIZE)
print("Learning rate :", LEARNING_RATE)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    train_dice = 0.0

    train_iou = 0.0

    for batch_idx, (images, masks) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        # Remise à zéro des gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss
        loss = loss_fn(
            outputs,
            masks
        )

        # Backpropagation
        loss.backward()

        # Mise à jour des paramètres
        optimizer.step()

        train_loss += loss.item()

        train_dice += dice_score(
            outputs.detach(),
            masks
        )

        train_iou += iou_score(
            outputs.detach(),
            masks
        )

        # Affichage de progression
        if (batch_idx + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"- Batch {batch_idx + 1}/{len(train_loader)} "
                f"- Loss : {loss.item():.4f}"
            )

    # Moyennes
    train_loss /= len(train_loader)

    train_dice /= len(train_loader)

    train_iou /= len(train_loader)


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_dice = 0.0

    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = loss_fn(
                outputs,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                outputs,
                masks
            )

            val_iou += iou_score(
                outputs,
                masks
            )

    # Moyennes
    val_loss /= len(val_loader)

    val_dice /= len(val_loader)

    val_iou /= len(val_loader)


    # --------------------------------------------------------
    # RESULTATS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Dice : {train_dice:.4f}"
    )

    print(
        f"Train IoU  : {train_iou:.4f}"
    )

    print()

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Dice   : {val_dice:.4f}"
    )

    print(
        f"Val IoU    : {val_iou:.4f}"
    )


    # --------------------------------------------------------
    # SAUVEGARDE DU MEILLEUR MODELE
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model_path = os.path.join(
            MODEL_DIR,
            "best_model.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou
            },
            model_path
        )

        print()
        print(
            "✓ Meilleur modèle sauvegardé :",
            model_path
        )

    print()
