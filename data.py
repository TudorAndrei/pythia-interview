import os
import pretty_errors
import pandas as pd
from typing import Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaTokenizerFast
from transformers.models.auto.tokenization_auto import AutoTokenizer
from sklearn.model_selection import train_test_split
from utils import TYPES, TYPES_DICT

class BaseDataset(Dataset):
    def __init__(
        self,
        X_y,
        tokenizer_path: Optional[str] = None,
    ):
        super().__init__()
        self.X, self.y = X_y
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            tokenizer_path
        )

    def get_label_array(self, idx):
        array = []
        for i, col in enumerate(TYPES):
            bin = self.y.iloc[idx, i]
            array.append(TYPES_DICT[col][bin])

        return array

    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.get_label_array(idx)
        text = self.tokenizer.encode_plus(
            text,
            # add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
        )
        return {
            "ids": torch.tensor(text["input_ids"], dtype=torch.long),
            # "mask": torch.tensor(text["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.float),
        }

    def __len__(self):
        return len(self.y)


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer_path: str = None,
        num_workers: int = 1,
        batch_size: int = 1,
        model_name: str = None,
    ):
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.tokenizer_path = tokenizer_path
        self.data = pd.read_csv(data_path)

    def setup(self, stage: Optional[str] = None) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['posts'], self.data[TYPES], stratify=self.data[TYPES])
        self.train_dataset = BaseDataset((self.X_train.values, self.y_train.reset_index(drop=True)), self.tokenizer_path)
        self.test_dataset = BaseDataset((self.X_test.values, self.y_test.reset_index(drop=True)), self.tokenizer_path)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
