from Bio import SeqIO
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from typing import Union, Optional, Tuple
from pathlib import Path
import tarfile
import pandas as pd
import numpy as np
from protscape.utils import seq_to_ohe, ohe_to_seq, download_tape, download_envision


class TAPEDataset(Dataset):
    def __init__(self, json_path: Union[str, Path], dataset: str, pad: int) -> None:

        if dataset == "stability":
            target = "stability_score"
        elif dataset == "fluorescence":
            target = "log_fluorescence"

        self.df = pd.read_json(json_path)
        self.seqs = self.df["primary"].values

        self.X = np.array([seq_to_ohe(seq, pad=pad) for seq in self.seqs])
        self.y = self.df[target].apply(lambda x: x[0]).values

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        return self.X[idx], self.y[idx]


class TAPEDataModule(pl.LightningDataModule):
    def __init__(
            self, dataset: str, pad: int, data_dir: Union[str, Path] = "./data", batch_size: int = 64,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.pad = pad
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

    def prepare_data(self) -> None:

        """Download TAPE json dataset."""

        download_tape(dataset=self.dataset, data_dir=self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:

        """Setup datasets for train, valid, and test splits."""

        if stage == "fit" or stage is None:
            json_path = Path(
                f"{self.data_dir}/{self.dataset}/{self.dataset}_train.json"
            )
            self.tape_train = TAPEDataset(json_path=json_path, dataset=self.dataset, pad=self.pad)

        if stage == "validation" or stage is None:
            json_path = Path(
                f"{self.data_dir}/{self.dataset}/{self.dataset}_valid.json"
            )
            self.tape_valid = TAPEDataset(json_path=json_path, dataset=self.dataset, pad=self.pad)

        if stage == "test" or stage is None:
            json_path = Path(f"{self.data_dir}/{self.dataset}/{self.dataset}_test.json")
            self.tape_test = TAPEDataset(json_path=json_path, dataset=self.dataset, pad=self.pad)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.tape_train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.tape_valid, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.tape_test, batch_size=self.batch_size)


class EnvisionDataset(Dataset):
    def __init__(self, fasta_path: Union[str, Path] = "./data/P62593.fasta"):

        self.df = self.fasta_to_df(fasta_path)

        self.X = np.array([seq_to_ohe(seq) for seq in self.df["seq"]])
        self.y = self.df["activity"].values

    def fasta_to_df(self, fasta_path):

        # read fasta
        records = SeqIO.parse(fasta_path, "fasta")

        # get mutation and activity from record header, save seq
        data = []
        for record in records:
            seq = str(record.seq)
            _, sample, activity = record.name.split("|")
            _, mutation = sample.split("_")

            data.append([mutation, seq, activity])

        # to dataframe
        df = pd.DataFrame(data, columns=["mutation", "seq", "activity"])
        df["activity"] = df["activity"].apply(pd.to_numeric, errors="ignore")

        return df

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        return self.X[idx], self.y[idx]


class EnvisionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Union[str, Path], batch_size: int=64):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fasta_path = Path(data_dir, 'P62593.fasta')

    def prepare_data(self):

        download_envision(data_dir = self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:

        self.envision = EnvisionDataset(fasta_path=self.fasta_path)

        # 80 train/20 test split
        n_train = int(0.8 * len(self.envision))
        n_test = len(self.envision) - n_train

        self.envision_train, self.envision_test = random_split(dataset=self.envision, 
                                                               lengths=[n_train, n_test], 
                                                               generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return DataLoader(self.envision_train, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.envision_test, batch_size=self.batch_size)
