import os
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_2020 = "data/processed/images/2020"
IMAGES_2024 = "data/processed/images/2024"
MASKS = "data/processed/masks"

OUTPUT = "data/ml_dataset"

# Pour obtenir toujours le même découpage
random.seed(42)


# ============================================================
# CRÉATION DES DOSSIERS
# ============================================================

for split in ["train", "validation", "test"]:
    os.makedirs(f"{OUTPUT}/{split}/images_2020", exist_ok=True)
    os.makedirs(f"{OUTPUT}/{split}/images_2024", exist_ok=True)
    os.makedirs(f"{OUTPUT}/{split}/masks", exist_ok=True)


# ============================================================
# RECHERCHE DES COUPLES IMAGE 2020 + IMAGE 2024 + MASQUE
# ============================================================

couples = []

for filename in os.listdir(IMAGES_2020):

    # On ne garde que les fichiers TIFF
    if not filename.endswith(".tif"):
        continue

    # Exemple :
    # manaus_2020_x0_y0.tif
    #
    # devient :
    # x0_y0

    coordonnees = filename.replace(
        "manaus_2020_", ""
    ).replace(
        ".tif", ""
    )

    image_2024 = f"manaus_2024_{coordonnees}.tif"
    masque = f"mask_{coordonnees}.tif"

    chemin_2020 = os.path.join(IMAGES_2020, filename)
    chemin_2024 = os.path.join(IMAGES_2024, image_2024)
    chemin_masque = os.path.join(MASKS, masque)

    # Vérification que les trois fichiers existent
    if (
        os.path.exists(chemin_2020)
        and os.path.exists(chemin_2024)
        and os.path.exists(chemin_masque)
    ):
        couples.append(
            (
                chemin_2020,
                chemin_2024,
                chemin_masque,
                coordonnees
            )
        )


# ============================================================
# AFFICHAGE DU NOMBRE DE COUPLES
# ============================================================

print("=" * 60)
print("PRÉPARATION DU DATASET")
print("=" * 60)

print(f"Couples complets trouvés : {len(couples)}")


# ============================================================
# MÉLANGE DES DONNÉES
# ============================================================

random.shuffle(couples)


# ============================================================
# CALCUL TRAIN / VALIDATION / TEST
# ============================================================

nombre_total = len(couples)

nombre_train = int(nombre_total * 0.70)
nombre_validation = int(nombre_total * 0.15)

nombre_test = (
    nombre_total
    - nombre_train
    - nombre_validation
)


train = couples[:nombre_train]

validation = couples[
    nombre_train:
    nombre_train + nombre_validation
]

test = couples[
    nombre_train + nombre_validation:
]


print(f"Train       : {len(train)}")
print(f"Validation  : {len(validation)}")
print(f"Test        : {len(test)}")


# ============================================================
# FONCTION DE COPIE
# ============================================================

def copier_dataset(couples, split):

    for chemin_2020, chemin_2024, chemin_masque, coordonnees in couples:

        # Noms des fichiers de destination

        destination_2020 = (
            f"{OUTPUT}/{split}/images_2020/"
            f"manaus_2020_{coordonnees}.tif"
        )

        destination_2024 = (
            f"{OUTPUT}/{split}/images_2024/"
            f"manaus_2024_{coordonnees}.tif"
        )

        destination_masque = (
            f"{OUTPUT}/{split}/masks/"
            f"mask_{coordonnees}.tif"
        )

        # Copie des fichiers

        shutil.copy2(
            chemin_2020,
            destination_2020
        )

        shutil.copy2(
            chemin_2024,
            destination_2024
        )

        shutil.copy2(
            chemin_masque,
            destination_masque
        )


# ============================================================
# COPIE DES DONNÉES
# ============================================================

print("\nCopie des données...")

copier_dataset(train, "train")

print("Train terminé.")

copier_dataset(validation, "validation")

print("Validation terminée.")

copier_dataset(test, "test")

print("Test terminé.")


# ============================================================
# FIN
# ============================================================

print("\n" + "=" * 60)
print("DATASET TERMINÉ")
print("=" * 60)

print(f"Dataset créé dans : {OUTPUT}")
