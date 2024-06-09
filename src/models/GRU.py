import torch.nn as nn
import torch.nn.functional as F
from src.models.generalModels import *
from src.models.LightningModel import LitModel

from typing import Tuple
from torch import Tensor


class GRU(LitModel):

    def __init__(
        self,
        learning_rate,
        input_size=5,
        hidden_size=64,
        classes=36,
        batch_first=True,
        layers=1,
        dense_layer=(False, 64),
        dropout=0.2,
    ):
        super().__init__()

        # set hyperparameters
        self.lr = learning_rate
        self.dense = dense_layer
        self.classes = classes

        # if dense before RNN
        if self.dense:
            self.RNN = nn.Sequential(
                nn.Linear(input_size, 2 * hidden_size),
                nn.GRU(
                    2 * hidden_size,
                    hidden_size,
                    num_layers=layers,
                    batch_first=batch_first,
                ),
            )
        else:
            self.RNN = nn.GRU(
                input_size, hidden_size, num_layers=layers, batch_first=batch_first
            )

        self.output_layers = outputRNN(
            hidden_size=hidden_size, transformed_size=2*hidden_size, output_size=self.classes, dropout=dropout
        )

    def forward(
        self, x: Tensor, x_mask: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        hidden_states, outputs = self.RNN(
            x
        )  # hidden states of all cells, outputs of last cells
        logits = self.output_layers(
            outputs[-1, :, :]
        )  # output of last cell into dense layer
        loss = F.cross_entropy(logits, y_targets)  # cross entropy loss
        return logits, loss
