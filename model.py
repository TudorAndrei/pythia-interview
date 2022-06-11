import math

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.modules.sparse import Embedding
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.classification.f_beta import f1_score


class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout: float = 0.1, vocab_size: int = 5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(vocab_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(vocab_size, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        d_out: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.encoder = Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, ntoken)

        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # self.decoder = Linear(d_model * seq_len, 1)
        self.decoder = Linear(d_model, d_out)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # print(src.shape)
        output = self.transformer_encoder(src)
        # print(output.shape)
        # output = output.reshape(output.shape[0], -1)
        output = output.mean(dim=1)
        # print(output.shape)
        output = self.decoder(output)
        # print(output.shape)
        output = torch.squeeze(output)
        return output


class Transformer(LightningModule):
    def __init__(
        self,
        ntokens=30000,  # size of vocabulary
        emsize=128,  # embedding dimension
        d_hid=128,  # dimension of the feedforward network model in nn.TransformerEncoder
        nlayers=1,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=1,  # number of heads in nn.MultiheadAttention
        d_out=4,
        dropout=0.2,  # dropout probability
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerModel(
            ntokens, emsize, nhead, d_hid, nlayers, d_out, dropout
        )
        self.criterion = BCEWithLogitsLoss()
        self.lr = 0.03
        self.n_classes = d_out

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, threshold=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }

    def training_step(self, batch, _):
        ids, labels = batch["ids"], batch["labels"]
        output = self(ids)
        loss = self.criterion(output, labels)
        return {"loss": loss, "outputs": torch.sigmoid(output), "labels": labels}

    def training_epoch_end(self, out):
        loss = torch.stack([x["loss"] for x in out]).mean()
        output = torch.cat([x["outputs"] for x in out])
        labels = torch.cat([x["labels"] for x in out])
        f1 = f1_score(
            preds=output,
            target=labels.int(),
            average="macro",
            num_classes=self.n_classes,
        )
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/f1_score", f1, prog_bar=True, on_epoch=True, on_step=False)

    def validation_step(self, batch, _):
        ids, labels = batch["ids"], batch["labels"]
        output = self(ids)
        loss = self.criterion(output, labels)
        return {"loss": loss, "outputs": torch.sigmoid(output), "labels": labels}

    def validation_epoch_end(self, out):
        loss = torch.stack([x["loss"] for x in out]).mean()
        output = torch.cat([x["outputs"] for x in out])
        labels = torch.cat([x["labels"] for x in out])
        f1 = f1_score(
            preds=output,
            target=labels.int(),
            average="macro",
            num_classes=self.n_classes,
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/f1_score", f1, prog_bar=True, on_epoch=True, on_step=False)
