import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from protscape import datasets
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from typing import List, Dict, Tuple, Union


class MaxSeqPool(nn.Module):

    """Layer to apply torch.max over sequence dimension."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=-1)[0]


class CNNEncoder(pl.LightningModule):

    """Landscape prediction from one-hot protein sequences.

    Parameters 
    ----------
    learning_rates : List[float]
        List of learning rates for the encoder, mapper, and predictor components
    weight_decays : List[float]
        List of weight decay values for the encoder, mapper, and predictor components
    reduction : str
        Pooling operation to apply on CNN encodings 
    encoder_kwargs : Dict
        Options for CNN encoder

    Attributes
    ----------
    encoder : nn.Sequential
        Generally a 1-3 layers CNN that converts a one-hot sequence to embedding
    mapper : nn.Sequential
        Pooling operation, either maxpool (get max on sequence length dimension of embedding) 
        or linear_maxpool (linear layer then get max on seq dim). Results in fixed length output, 
        not a regular pooling operation since its not a sliding window on the sequence.
    predictor : nn.Sequential
        Single fully connected layer taking pooled embedding to activity prediction

    Steps
    -----
    End-to-end optimization of OHE seq -> CNN encoder -> mapper (pooling operation) -> predictor
    """

    def __init__(
        self,
        learning_rates: List,
        weight_decays: List,
        reduction: str,
        encoder_kwargs: Dict,
    ) -> None:

        super().__init__()

        self.learning_rates = learning_rates
        self.weight_decays = weight_decays
        self.reduction = reduction

        encoder_out = encoder_kwargs["out_channels"][-1]

        self.encoder = nn.Sequential(*self.make_encoder(encoder_kwargs))
        self.mapper = nn.Sequential(*self.make_mapper(reduction, encoder_out))
        self.predictor = nn.Sequential(*self.make_predictor(reduction))

    def make_encoder(self, encoder_kwargs: Dict) -> List:

        """From one-hot to CNN encoded representation."""

        conv_layers = encoder_kwargs["layers"]
        out_channels = encoder_kwargs["out_channels"]
        kernel_sizes = encoder_kwargs["kernel_sizes"]
        dilations = encoder_kwargs["dilations"]

        encoder_layers = []

        for i in range(conv_layers):

            # 22 valid characters (20 amino acids + gap + unknown)
            in_chan = 22 if i == 0 else out_channels[i - 1]

            out_chan = out_channels[i]
            kernel_size = kernel_sizes[i]
            dilation = dilations[i]

            # same padding (kernel -1)/2
            pad = kernel_size // 2

            layer = nn.Conv1d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilation,
            )

            encoder_layers.append(layer)
            encoder_layers.append(nn.ReLU(inplace=True))

        return encoder_layers

    def make_mapper(self, reduction: str, in_chan: int) -> List:

        """Pool the CNN encoder output."""

        mapper_layers = []

        if reduction == "linear_maxpool":
            layer = nn.Linear(in_chan, 2048)
            mapper_layers.append(layer)
            mapper_layers.append(nn.ReLU(inplace=True))

        mapper_layers.append(MaxSeqPool())

        return mapper_layers

    def make_predictor(self, reduction: str) -> List:

        """Make activity prediction from embedding."""

        predictor_layers = []

        # maxpool out is 1024, linear_maxpool out is 2048
        in_chan = 2048 if reduction == "linear_maxpool" else 1024

        layer = nn.Linear(in_chan, 1)
        predictor_layers.append(layer)

        return predictor_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)
        x = self.mapper(x)
        x = self.predictor(x)

        return x

    def configure_optimizers(self):

        # set separate learning rates/weight decays for the encoder, mapper, and predictor
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.encoder.parameters(),
                    "lr": self.learning_rates[0],
                    "weight_decay": self.weight_decays[0],
                },
                {
                    "params": self.mapper.parameters(),
                    "lr": self.learning_rates[1],
                    "weight_decay": self.weight_decays[1],
                },
                {
                    "params": self.predictor.parameters(),
                    "lr": self.learning_rates[2],
                    "weight_decay": self.weight_decays[2],
                },
            ]
        )

        return optimizer

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        x, y = train_batch

        # from (batch, seq length, charset) -> (batch, charset, seq length)
        x = x.permute(0, 2, 1).float()
        y = y.unsqueeze(-1).float()

        preds = self(x)

        loss = F.mse_loss(preds, y)
        self.log("train_loss", loss)

        return loss

    def test_step(
        self, test_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        x, y = test_batch
        x = x.permute(0, 2, 1).float()
        y = y.unsqueeze(-1).float()

        preds = self(x)

        loss = F.mse_loss(preds, y)
        self.log("test_loss", loss)

        return loss

    def run_predict(self, loader: DataLoader) -> Tuple[List, List]:

        """Predict activity on input dataloader."""

        self.eval()

        preds = []
        true = []

        with torch.no_grad():
            for batch in loader:
                x, y = batch

                x = x.permute(0, 2, 1).float()

                pred = self(x).reshape(-1)

                preds.extend(pred.detach().cpu().numpy())
                true.extend(y.numpy())

        return preds, true
