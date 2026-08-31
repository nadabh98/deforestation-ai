import torch
from torch.utils.data import DataLoader
from dataset import DeforestationDataset


# ============================================================
# CHARGEMENT DU DATASET
# ============================================================

dataset = DeforestationDataset(
    "data/ml_dataset/train"
)


# ============================================================
# CREATION DU DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)


# ============================================================
# RECUPERATION D'UN BATCH
# ============================================================

images, masks = next(iter(loader))


# ============================================================
# AFFICHAGE
# ============================================================

print("=" * 60)
print("TEST DU DATALOADER")
print("=" * 60)

print("Images :")
print("Shape :", images.shape)
print("Type  :", images.dtype)

print()

print("Masques :")
print("Shape :", masks.shape)
print("Type  :", masks.dtype)

print()

print("Valeurs des masques :")
print(torch.unique(masks))

print()

print("Min image :", images.min().item())
print("Max image :", images.max().item())

print("=" * 60)
