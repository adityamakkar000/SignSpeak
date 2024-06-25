import torch.nn as nn
import torch.nn.functional as F
from src.models.generalModels import outputRNN
from src.models.LightningModel import LitModel

from typing import Tuple
from torch import Tensor


class LSTM(LitModel):

    def __init__(
        self,
        learning_rate,
        input_size=5,
        hidden_size=64,
        classes=36,
        batch_first=True,
        layers=1,
        dense_layer= False,
        dropout=0.2,
    ):
        super().__init__()

        # set hyperparameters
        self.dense = dense_layer
        self.classes = classes
        self.lr = learning_rate

        # if dense before RNN
        if self.dense:
            self.RNN = nn.Sequential(
                nn.Linear(input_size, 2 * hidden_size),
                nn.LSTM(
                    2 * hidden_size,
                    hidden_size,
                    num_layers=layers,
                    batch_first=batch_first,
                ),
            )
        else:
            self.RNN = nn.LSTM(
                input_size, hidden_size, num_layers=layers, batch_first=batch_first
            )

        self.output_layers = outputRNN(
            hidden_size=hidden_size, transformed_size=2*hidden_size ,output_size=self.classes, dropout=dropout
        )

    def forward(
        self, x: Tensor, x_mask: Tensor, y_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        hidden_states, (outputs, cell_states) = self.RNN(
            x
        )  # hidden states of all cells, outputs of last cells
        logits = self.output_layers(outputs[-1, :, :])  # outputs (batch, classes)
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss
