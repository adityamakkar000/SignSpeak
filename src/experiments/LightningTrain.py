# for local machines
import sys
import argparse
import os
import os.path as op

path = op.dirname(op.dirname(op.dirname(op.realpath(__file__))))
sys.path.append(path)

# !pip install torcheval # for colab
# !pip install wandb
# !pip install Lightning

import numpy as np
import random
import time
import wandb
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import DeviceStatsMonitor

# local imports
from src.models.encoder import Encoder
from src.models.encoder import Encoder_self_weighting
from src.models.encoder import EncoderCOVN
from src.models.LSTM import LSTM
from src.models.GRU import GRU
from src.misc.DataModule import ASLDataModule

# set determinstic behaviour
# torch.use_deterministic_algorithms(True) # don't use on GPU

# setup cli arg parser
parser = argparse.ArgumentParser(description="model traning")
parser.add_argument(
    "-description", dest="description", type=str, required=False
)  # description for wandb

parser.add_argument(
    "-test", dest="test", action="store_true", required=False
)  # train mode


# general params
parser.add_argument(
    "-time_steps", dest="time_steps", type=int, required=False
)  # time steps
parser.add_argument("-lr", dest="lr", type=float, required=False)  # learning rate
parser.add_argument(
    "-batch_size", dest="batch_size", type=int, required=False
)  # batch size
parser.add_argument("-epochs", dest="epochs", type=int, required=False)  # epochs

parser.add_argument(
    "-model", dest="model", type=str, required=False
)  # get type of model
parser.add_argument(
    "-layers", dest="layers", type=int, required=False
)  # number of layers
parser.add_argument(
    "-hidden_size", dest="hidden_size", type=int, required=False
)  # hidden size

# RNN params
parser.add_argument(
    "-dense_layer", dest="dense_layer", action="store_true", required=False
)  # dense layer

# encoder params
parser.add_argument(
    "-number_heads", dest="number_heads", type=int, required=False
)  # number of heads

parser.add_argument(
    "-project_name", dest="project_name", type=str, help="project name for wandb logging", required=False
)

parser


parser.set_defaults(
    dense_layer=False,
    batch_size=64,
    epochs=600,
    time_steps=79,
    lr=1e-4,
    model="Encoder",
    layers=1,
    hidden_size=5,
    number_heads=1,
    project_name="SignSpeak",
    test=False
)

args = parser.parse_args()


# hyper parameters
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr
time_steps = args.time_steps
n_emb = 5
classes = 36
seed = 1337

# set seeds
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# set model params for testing
encoder_params = {
    "layers": args.layers,
    "time_steps": time_steps,
    "number_heads": args.number_heads,
    "input_size": 5,
    "classes": classes,
    "hidden_size": args.hidden_size,
    "learning_rate": learning_rate,
}


RNN_params = {
    "input_size": 5,
    "layers": args.layers,
    "dense_layer": True if args.dense_layer else False,
    "hidden_size": args.hidden_size,
    "classes": classes,
    "learning_rate": learning_rate,
}

params = {"Encoder": encoder_params,
          "LSTM": RNN_params,
          "GRU": RNN_params,
          "EncoderSW": encoder_params,
          "EncoderCONV": encoder_params}


def get_model(t, params):
    """take model type and return that with the desired parameters"""
    model_types = {"LSTM": LSTM, "GRU": GRU, "Encoder": Encoder, "EncoderSW": Encoder_self_weighting, "EncoderCONV": EncoderCOVN}
    return model_types[t](**params[t])


# call models and check layers
type_of_model = args.model
model = get_model(type_of_model, params)
model.info(layers=True)
print(model.total_params())


splits = 5  # k-fold splits
project_name = args.project_name
wandb_log = True if args.description else False
print(wandb_log, " ", args)

# k-fold number
for split_number in range(splits):

    # ensure seed is the same
    torch.manual_seed(seed)
    model = get_model(type_of_model, params)  # intialize model
    if wandb_log:
        run_name = (
            type_of_model
            + "_"
            + args.description
            + "_"
            + str(split_number + 1)
            + "-fold_"
            + datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        )
        wandb_logger = WandbLogger(
            project=project_name, name=run_name
        )  # setup wandb logger
        config = {
            "learning_rate": learning_rate,
            "context length": time_steps,
            "total parameters": model.total_params(),
            "params": params,
            "classes": classes,
            "seed": seed,
            "epochs": epochs,
            "kfold-split": split_number + 1,
        }
        wandb_logger.experiment.config.update(config)

    pf = AdvancedProfiler(dirpath="./", filename="profile.txt")
    trainer_params = {"max_epochs": epochs, "log_every_n_steps": 1, "fast_dev_run": True if args.test else False}

    dataset_params = {
        "n_emb": n_emb,
        "time_steps": time_steps,
        "kfold": split_number,
        "splits": splits,
        "seed": seed,
        "batch_size": batch_size,
        "shuffle": True,
    }

    # intialize dataset, traniner and fit model
    dataset = ASLDataModule(**dataset_params)
    trainer = (
        L.Trainer(**trainer_params, logger=wandb_logger)
        if wandb_log
        else L.Trainer(**trainer_params)
    )
    trainer.fit(model, dataset)

    if wandb_log:
        wandb.finish()
