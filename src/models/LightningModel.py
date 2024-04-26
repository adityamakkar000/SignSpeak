import numpy as np
import torch
import lightning as L
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix

from src.models.generalModels import ModelInfo

from typing import Tuple, Dict
from torch import Tensor


class LitModel(L.LightningModule, ModelInfo):


  def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
    """ training step for the model """

    x,y = batch

    logits, loss = self(batch[x],batch[y]) # forward pass
    self.log('training-loss', loss)

    return loss

  def configure_optimizers(self):

    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    # TODO: Add scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer

  def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

    x,y = batch

    logits, loss =  self(batch[x],batch[y]) # forward pass
    logits_argmax = torch.argmax(logits, dim=-1) # get argmax of logits
    # confusion matrix
    cm = confusion_matrix(batch[y].cpu(),
                            logits.cpu().argmax(axis=1).numpy(), labels=np.arange(self.classes).tolist())
    true_acc, cat_acc = self.get_accuracy(cm) # call accuracy from misc.py
    val_f1 = multiclass_f1_score(logits, batch[y],num_classes=self.classes, average=None) # f1-score

    # log metrics
    self.log("val-loss", loss)
    self.log('true accuracy', torch.tensor(true_acc).mean().item())
    self.log('categorical accuracy', torch.tensor(cat_acc).mean().item())
    self.log('f1-score', val_f1.mean().item())

    return loss
