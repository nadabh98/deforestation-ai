import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss pour la segmentation binaire.
    """

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth

    def forward(self, logits, targets):

        # Conversion des logits en probabilités
        probabilities = torch.sigmoid(logits)

        # Aplatissement
        probabilities = probabilities.view(
            probabilities.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        # Intersection
        intersection = (
            probabilities * targets
        ).sum(dim=1)

        # Score Dice
        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        # Dice Loss
        loss = 1.0 - dice

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combinaison BCE + Dice.
    """

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets.unsqueeze(1)
        )

        dice_loss = self.dice(
            logits,
            targets.unsqueeze(1)
        )

        total_loss = (
            bce_loss + dice_loss
        )

        return total_loss
