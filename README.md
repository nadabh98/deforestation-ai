# 🌳 Deforestation AI — Détection de la déforestation par satellite

## 📌 Présentation

Deforestation AI est un projet de Machine Learning permettant de détecter automatiquement les zones potentiellement déforestées à partir d'images satellites Sentinel-2.

Le projet compare des images satellite de **2020 et 2024** afin d'identifier les changements pouvant correspondre à de la déforestation.

Le projet s'inscrit dans le cadre de l'**ODD 15 — Vie terrestre**, qui vise notamment à protéger les écosystèmes terrestres et à lutter contre la dégradation des forêts.

---

## 🎯 Objectifs

Les objectifs du projet sont :

- récupérer des images satellites Sentinel-2 ;
- préparer et nettoyer les données géospatiales ;
- construire un dataset Machine Learning ;
- entraîner un modèle de segmentation U-Net ;
- détecter les zones de déforestation ;
- évaluer les performances du modèle ;
- exposer le modèle via une API REST ;
- conteneuriser l'application avec Docker ;
- déployer l'API sur Google Cloud Run ;
- créer un dashboard interactif avec Streamlit ;
- visualiser les résultats sur une carte.

---

## 🛰️ Données

Les données utilisées proviennent de Sentinel-2.

Pour chaque zone étudiée, quatre bandes spectrales sont utilisées :

| Bande | Description |
|---|---|
| B2 | Bleu |
| B3 | Vert |
| B4 | Rouge |
| B8 | Proche infrarouge (NIR) |

Le modèle reçoit donc :

- 4 bandes pour 2020
- 4 bandes pour 2024

soit **8 canaux en entrée**.

Les images sont découpées en tuiles de :

**256 × 256 pixels**

---

## 🧠 Modèle Machine Learning

Le modèle utilisé est un réseau de segmentation sémantique basé sur une architecture **U-Net**.

### 🧠 Méthodologie

Le pipeline global du projet est le suivant :

                 Sentinel-2
                     │
                     ▼
          Acquisition des images
                     │
                     ▼
          Prétraitement des données
                     │
                     ▼
        Création des masques binaires
                     │
                     ▼
              Dataset ML
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        Train       Val        Test
          │
          ▼
       U-Net
          │
          ▼
       Entraînement
          │
          ▼
       Évaluation
          │
          ▼
    Choix du seuil = 0.70
          │
          ▼
     Prédiction 2020→2024
          │
          ▼
       FastAPI
          │
          ▼
       Cloud Run
          │
          ▼
      Dashboard

## 🗃️ Préparation du dataset:

Les images satellites ont été découpées en tuiles de :

256 × 256 pixels

Le dataset final contient :

Train       : 196 exemples
Validation  : 42 exemples
Test        : 42 exemples
Total       : 280 exemples

Répartition :

70 % → Train
15 % → Validation
15 % → Test

Chaque exemple contient :

image_2020
image_2024
mask

## 📊 Évaluation du modèle

L'évaluation finale a été réalisée sur le dataset de test.

Le seuil de classification utilisé est :

Threshold = 0.70

Résultats obtenus :

Métrique	Résultat
Dice	0.8582
IoU	0.7517
Precision	0.8863
Recall	0.8319

Les valeurs correspondantes sont :

True Positives  : 104 980
False Positives : 13 467
False Negatives : 21 216

Ces résultats montrent que le modèle est capable d'identifier une grande partie des zones de déforestation présentes dans les données de test.

##📈 9. Résultats du modèle

L'évaluation finale a été effectuée sur le jeu de test composé de :

42 exemples

Avec un seuil de décision de :

0.70

Résultats :

Métrique	Résultat
Dice	0.8582
IoU	0.7517
Precision	0.8863
Recall	0.8319

Nombre de pixels :

TP : 104 980
FP : 13 467
FN : 21 216

Ces résultats montrent que le modèle est capable d'identifier une grande partie des zones de déforestation tout en limitant les faux positifs.
