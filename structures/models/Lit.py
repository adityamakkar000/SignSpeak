import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class LitModel(L.LightningModule):

  def training_step(self, batch, batch_idx):
    x,y = batch
    logits, loss = self(batch[x],batch[y])

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    return optimizer

  def validation_step(self, batch, batch_idx):
    x,y = batch
    logits, loss =  self(batch[x],batch[y])

    return loss
