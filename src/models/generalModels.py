import torch.nn as nn
import numpy as np

from typing import List, Tuple
from torch import Tensor


class ModelInfo:

    def get_accuracy(self, cm: List[List[int]]) -> Tuple[List[float], List[float]]:
        """Get accuracy from confusion matrix"""
        TP = np.diag(cm)  # True Positives are the sume of the diagonal
        # True Negatives are the sum of all elements except the given row and column
        TN = np.sum(np.sum(cm, axis=0)) - (
            np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm)
        )
       


        FP = np.sum(cm, axis=0) - np.diag(
            cm
        )  # False Positives are the sum of the column minus the diagonal
        FN = np.sum(cm, axis=1) - np.diag(
            cm
        )  # False Negatives are the sum of the row minus the diagonal
        true_acc = (TP + TN) / (TP + TN + FP + FN)  # True Accuracy (1 x self.classes)
        cat_acc = TP / (FN + TP)  # Categorical Accuracy (1 x self.classes)
        return true_acc, cat_acc

    def info(self, layers=False):
        """Print model information"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        if layers:
            print("Layers:")
            print(self.named_parameters())
            for name, param in self.named_parameters():
                print(
                    f"Layer: {name}, Size: {param.size()}, Parameters: {param.numel()}, Device: {param.device}"
                )

    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class outputRNN(nn.Module):

    def __init__(
        self, hidden_size=64, transformed_size=64, output_size=10, dropout=0.2
    ):
        """Output layer for RNN models"""
        super().__init__()
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, transformed_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(transformed_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass"""
        logits = self.output_layers(x)  # (batch, classes)
        return logits
