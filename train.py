import glob
import logging as log
import os

import pretty_errors
import wandb
from dotenv import dotenv_values
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import logging
from whos_there.callback import NotificationCallback
from whos_there.senders.discord import DiscordSender

from data import DataModule
from model import Transformer

logging.set_verbosity_error()
config = dotenv_values(".env")

log.getLogger("pytorch_lightning").setLevel(log.WARNING)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed_everything(42)

BATCH_SIZE = 32
NW = 16
EPOCHS = 100
web_hook = config["DISCORD_WEBHOOK"]

sweep_config = {
    "method": "grid",
    "metric": {"name": "val/f1_score", "goal": "minimize"},
    "parameters": {
        "emsize": {"values": [256]},
        "d_hid": {"values": [128]},
        "nlayers": {"values": [3]},
        "nhead": {"values": [4]},
    },
}


def sweep_iteration():
    model_name = "transformer"
    wandb.init()
    logger = WandbLogger()
    params = {
        "batch_size": BATCH_SIZE,
        "num_workers": NW,
        "model_name": model_name,
        "tokenizer_path": "models/tokenizer",
    }
    model_params = {
        "ntokens": 30000,
        "emsize": wandb.config.emsize,
        "d_hid": wandb.config.d_hid,
        "nlayers": wandb.config.nlayers,
        "nhead": wandb.config.nhead,
        "dropout": 0.2,
    }
    dm = DataModule(data_path="mbti_processed.csv", **params)
    version = len(glob.glob("models/trans*"))
    wandb.run.name = f"{model_name}_{version}"
    model = Transformer(**model_params)
    trainer = Trainer(
        # fast_dev_run=True,
        detect_anomaly=True,
        gpus=1,
        logger=logger,
        max_epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                dirpath=f"models/{model_name}_{version}",
                filename="model-{epoch:02d}-{val/loss:.2f}",
                auto_insert_metric_name=False,
            ),
            NotificationCallback(senders=[DiscordSender(webhook_url=web_hook)]),
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val/loss", patience=15),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="erisk")
    wandb.agent(sweep_id, function=sweep_iteration)
