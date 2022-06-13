import glob
import subprocess
import logging as log
import os

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
NW = int(subprocess.check_output(['nproc', '--all']))
EPOCHS = 100
web_hook = config["DISCORD"]

model_name = "transformer"
params = {
    "batch_size": BATCH_SIZE,
    "num_workers": NW,
    "model_name": model_name,
    "tokenizer_path": "models/tokenizer",
    # "tokenizer_path": model_name,
}

model_params = {
    "ntokens": 30000,
    "emsize": 128,
    "d_hid": 128,
    "nlayers": 2,
    "nhead": 2,
    "dropout": 0.2,
}


def loop():
    wandb.init()
    logger = WandbLogger(save_dir="logs", project=PROJECT)
    dm = DataModule(data_path="data/mbti_processed.csv", **params)
    version = len(glob.glob(f"models/{model_name}*"))
    model = Transformer(**model_params)
    trainer = Trainer(
        # fast_dev_run=True,
        detect_anomaly=True,
        gpus=1,
        logger=logger,
        max_epochs=EPOCHS,
        callbacks=[
            # ModelCheckpoint(
            #     monitor="val/loss",
            #     mode="min",
            #     dirpath=f"models/{model_name}_{version}",
            #     filename="model-{epoch:02d}-{val/loss:.2f}",
            #     auto_insert_metric_name=False,
            # ),
            NotificationCallback(senders=[DiscordSender(webhook_url=web_hook)]),
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/loss", patience=15),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    loop()
