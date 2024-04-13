import numpy as np
import torch
import lightning as L
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix

from misc import ModelInfo

class LitModel(L.LightningModule, ModelInfo):

  def training_step(self, batch, batch_idx):
    x,y = batch
    logits, loss = self(batch[x],batch[y])
    self.log('training-loss', loss)

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    return optimizer

  def validation_step(self, batch, batch_idx):
    x,y = batch
    logits, loss =  self(batch[x],batch[y])
    logits_argmax = torch.argmax(logits, dim=-1)
    cm = confusion_matrix(batch[y].cpu(),
                            logits.cpu().argmax(axis=1).numpy(), labels=np.arange(10).tolist())
    true_acc, cat_acc = self.get_accuracy(cm)
    val_f1 = multiclass_f1_score(logits, batch[y],num_classes=self.classes, average=None)

    self.log("val-loss", loss)
    self.log('true accuracy', torch.tensor(true_acc).mean().item())
    self.log('categorical accuracy', torch.tensor(cat_acc).mean().item())
    self.log('f1-score', val_f1.mean().item())

    return loss
