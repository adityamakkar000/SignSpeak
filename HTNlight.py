import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pack_sequence
import pytorch_lightning as pl
import pandas as pd
import csv
import numpy as np

class CustomImageDataset():
    def __init__(self):
        self.data = pd.read_csv("./cluster0.RNN_database.csv")

        lengths = []

        with open('./cluster0.RNN_database.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)

            selected_rows = []
            selected_rows_2 = []
            selected_columns = []
            count = 0

            for row in csvreader:

                if count == 0:
                    count += 1
                    continue
                else:
                    data = []
                    start = 0
                    while True:
                        resist_vals = []
                        for i in range(10):
                            try:
                                if row[2 + start + i] == '':
                                    break
                                resist_vals.append(float(row[2 + start + i]))
                            except IndexError:
                                break
                        if len(resist_vals) != 10:
                            break
                        data.append(resist_vals)
                        start += 10

                    lengths.append(len(data))
                    selected_columns = torch.Tensor(data)
                    selected_columns_2 = row[1]
                    selected_rows.append(selected_columns)
                    selected_rows_2.append(selected_columns_2)
                    count += 1
        # pad self.x from the selcted rows
        self.x = torch.nn.utils.rnn.pack_padded_sequence(torch.nn.utils.rnn.pad_sequence(selected_rows, batch_first=True, padding_value=0), lengths, batch_first=True, enforce_sorted=False).data
        self.y = selected_rows_2

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        return x, y

modified_data = CustomImageDataset()

dataloader = torch.utils.data.DataLoader(modified_data, batch_size=1, shuffle=False)

print(modified_data.x.__len__())
print(modified_data.y.__len__())

encoder = nn.RNN(10, 64, 2)
decoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(64, 15), nn.Softmax())


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):

        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


autoencoder = LitAutoEncoder(encoder, decoder)
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=30, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=dataloader)

# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

