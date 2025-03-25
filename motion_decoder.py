import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional
from torch import nn, Tensor

from utils import PositionalEncoding

def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

class MotionDecoder(pl.LightningModule):
    def __init__(self, nfeats: int, max_len: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 6, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout, max_len=max_len)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats

        z = z[None]  # sequence of 1 element for the memory

        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=time_queries, memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        return feats