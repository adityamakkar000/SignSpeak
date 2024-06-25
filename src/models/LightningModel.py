import numpy as np
import torch
import lightning as L
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix

from src.models.generalModels import ModelInfo

from typing import Tuple, Dict
from torch import Tensor


class LitModel(L.LightningModule, ModelInfo):

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """training step for the model"""

        x, x_padding, y = batch

        logits, loss = self(batch[x], batch[x_padding], batch[y])  # forward pass
        self.log("training-loss", loss)

        return loss

    def configure_optimizers(self):

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',factor=0.5, min_lr=0.0001, patience=20
        )

        optim = {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val-loss"

        }

        return optim

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        x, x_padding, y = batch

        logits, loss = self(batch[x], batch[x_padding], batch[y])  # forward pass
        logits_argmax = torch.argmax(logits, dim=-1)  # get argmax of logits
        # confusion matrix
        cm = confusion_matrix(
            batch[y].cpu(),
            logits.cpu().argmax(axis=1).numpy(),
            labels=np.arange(self.classes).tolist(),
        )
        true_acc, cat_acc = self.get_accuracy(cm)  # call accuracy from misc.py
        val_f1 = multiclass_f1_score(
            logits, batch[y], num_classes=self.classes, average=None
        )  # f1-score

        # log metrics
        self.log("val-loss", loss, prog_bar=True)
        self.log("categorical accuracy", torch.tensor(cat_acc).mean().item(), prog_bar=True)
        self.log("f1-score", val_f1.mean().item(), prog_bar=True)
        self.log("learning_rate", self.optimizer.param_groups[0]['lr'], prog_bar=True)

        if isinstance(self.logger, WandbLogger):
            plot = wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.array(batch[y].cpu()),
                preds=np.array(logits_argmax.cpu()),
                class_names=np.arange(self.classes).tolist(),
            )
            self.logger.experiment.log({"confusion_matrix": plot})

        val_loss = loss
        return val_loss
