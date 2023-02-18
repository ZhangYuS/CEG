import linecache
import random

import torch
import os
import json
import logging
import csv
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from argparse import Namespace
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from argparse import ArgumentParser

class KgDataloader(pl.LightningDataModule):

    def __init__(self, dataset_class, data_dir, train_batch_size, eval_batch_size, num_workers, **kwargs):
        super().__init__()
        self.dataset_class = dataset_class
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage=None):

        self.train_dataset = self.dataset_class(file_path=os.path.join(self.data_dir, 'train.txt'), stage='train', **self.kwargs)
        self.dev_dataset = self.dataset_class(file_path=os.path.join(self.data_dir, 'dev.txt'), stage='dev', **self.kwargs)
        self.dev_predict_dataset = self.dataset_class(file_path=os.path.join(self.data_dir, 'dev.txt'), stage='test', **self.kwargs)
        self.test_dataset = self.dataset_class(file_path=os.path.join(self.data_dir, 'test.txt'), stage='test', **self.kwargs)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.train_dataset.collact_fn, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.dev_dataset.collact_fn, persistent_workers=False)

    def val_predict_dataloader(self):
        return DataLoader(self.dev_predict_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.dev_predict_dataset.collact_fn, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.test_dataset.collact_fn, persistent_workers=False)

    def predict_dataloader(self):
        return self.test_dataloader()

    def ground_true(self):
        return self.test_dataset.ground_true()

    def dev_ground_true(self):
        return self.dev_dataset.ground_true()

    @staticmethod
    def add_dataloader_specific_args(parser: ArgumentParser):

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="path of the dataset"
        )


        parser.add_argument("--train_batch_size", default=10, type=int)
        parser.add_argument("--eval_batch_size", default=10, type=int)
        parser.add_argument("--num_workers", default=0, type=int, help="kwarg passed to DataLoader")

        return parser