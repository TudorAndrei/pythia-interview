import glob
import logging as log
import os
import subprocess

import pretty_errors
from dotenv import dotenv_values
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import logging
from whos_there.callback import NotificationCallback
from whos_there.senders.discord import DiscordSender

import wandb
from data import DataModule
from model import Transformer

logging.set_verbosity_error()
config = dotenv_values(".env")


log.getLogger("pytorch_lightning").setLevel(log.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed_everything(42)

PROJECT = "pythia-interview"
BATCH_SIZE = 128
NW = int(subprocess.check_output(["nproc", "--all"]))
EPOCHS = 100
WEB_HOOK = config["DISCORD"]
MODEL_NAME = "transformer"

sweep_config = {
    "method": "grid",
    "metric": {"name": "val/f1_score", "goal": "maximize"},
    "parameters": {
        "emsize": {"values": [32, 64, 128]},
        "d_hid": {"values": [32, 64, 128]},
        "nlayers": {"values": [1, 2]},
        "nhead": {"values": [1, 2, 4]},
    },
}

params = {
    "batch_size": BATCH_SIZE,
    "num_workers": NW,
    "model_name": MODEL_NAME,
    "tokenizer_path": "models/tokenizer",
}

def sweep_iteration():
    wandb.init()

    model_params = {
        "ntokens": 30000,
        "emsize": wandb.config.emsize,
        "d_hid": wandb.config.d_hid,
        "nlayers": wandb.config.nlayers,
        "nhead": wandb.config.nhead,
        "dropout": 0.2,
    }
    logger = WandbLogger(save_dir="logs", project=PROJECT)

    dm = DataModule(data_path="data/mbti_processed.csv", **params)
    version = len(glob.glob(f"models/{MODEL_NAME}*"))
    wandb.run.name = f"{MODEL_NAME}_{version}"
    model = Transformer(**model_params)
    trainer = Trainer(
        # fast_dev_run=True,
        detect_anomaly=True,
        gpus=1,
        logger=logger,
        max_epochs=EPOCHS,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/loss", patience=15),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    # sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent("jstpb4ql", project=PROJECT, function=sweep_iteration)
