import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from misc import ModelInfo, outputRNN
from Lit import LitModel

class LSTM(ModelInfo, LitModel):

    def __init__(self,
                 learning_rate,
                 input_size=5,
                 hidden_size=64,
                 output_size=10,
                 batch_first=True,
                 layers=1,
                 dense_layer=(False, 64),
                 dropout=0.2):
        super().__init__()
        self.dense, self.dense_size = dense_layer
        self.output_size = output_size
        self.lr = learning_rate

        if self.dense:
            self.RNN = nn.Sequential(
                nn.Linear(input_size, self.dense_size),
                nn.LSTM(self.dense_size,
                        hidden_size,
                        num_layers=layers,
                        batch_first=batch_first)
            )
        else:
            self.RNN = nn.LSTM(input_size,
                               hidden_size,
                               num_layers=layers,
                               batch_first=batch_first)

        self.output_layers = outputRNN(hidden_size=hidden_size,
                                       output_size=output_size,
                                       dropout=dropout)

    def forward(self, x, y_targets):
        hidden_states, (outputs, cell_states) = self.RNN(x)
        logits = self.output_layers(outputs[-1, :, :])
        logits = logits.view(-1, self.output_size)
        loss = F.cross_entropy(logits, y_targets)
        return logits, loss
