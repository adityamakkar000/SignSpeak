import os
from torch import optim, nn, utils, Tensor
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import pandas as pd
import csv
import numpy as np


# define any number of nn.Modules (or use your current ones)
encoder = nn.RNN(10, 64, 2)
decoder = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(64, 15), nn.Softmax())

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)

data = pd.read_csv("/Users/Aarav/Downloads/cluster0.RNN_database.csv")
print(data)


class CustomImageDataset(data):
    def __init__(self):
        self.data = pd.read_csv("/Users/Aarav/Downloads/cluster0.RNN_database.csv")

        with open('Users/Aarav/Downloads/cluster0.RNN_database.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
        
            selected_rows = []
            selected_rows_2 = []
        
            for row in csvreader:
                selected_columns = [row[2], row[11]]
                selected_columns_2 = [row[1]]
                selected_rows.append(selected_columns)
                selected_rows_2.append(selected_columns_2)
        selected_array_y = np.array(selected_rows)
        selected_array_x = np.array(selected_rows_2)
        self.y = [selected_array_y]
        self.x = [selected_array_x]
        print(self.y)
        print(" ")
        print(self.x)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y = torch.Tensor(self.y[idx])
        x = torch.Tensor(self.x[idx])
        return x, y
    
modified_data = CustomImageDataset(data)
    
dataloader = torch.utils.data.DataLoader(modified_data, batch_size=1, shuffle=False)

for batch in dataloader:
    print(batch)

exit()

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=30, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=data)

# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

