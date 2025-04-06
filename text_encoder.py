import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from typing import List, Union
from torch import Tensor
from torch.distributions.distribution import Distribution
from utils.positional_encoding import PositionalEncoding


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model: str = "distilbert-base-uncased",
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 384,
                 ff_size: int = 1024,
                 num_layers: int = 6, num_heads: int = 6,
                 dropout: float = 0.1,
                 activation: str = "gelu") -> None:
        super().__init__()

        # Load DistilBERT model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
        self.bert_model = DistilBertModel.from_pretrained(pretrained_model)
        if not finetune:
            print('Freezing BERT weights')
            for param in self.bert_model.parameters():
                param.requires_grad = False  # Freeze BERT weights

        encoded_dim = self.bert_model.config.hidden_size  # Typically 768 for DistilBERT

        # Projection of text embeddings into the latent space
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(encoded_dim, latent_dim)
        )

        # TransformerVAE structure
        self.vae = vae
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation
        )

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer, num_layers=num_layers)

    def forward(self, texts: List[str]) -> Union[Tensor, Distribution]:
        # Tokenize input text and send to device
        encodings = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(next(self.bert_model.parameters()).device)
        attention_mask = encodings["attention_mask"].to(input_ids.device)

        # Get BERT embeddings
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        text_encoded = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Project embeddings to latent space
        x = self.projection(text_encoded)  # (batch_size, seq_len, latent_dim)
        bs, nframes, _ = x.shape
        x = x.permute(1, 0, 2)  # Switch to [nframes, bs, latent_dim]

        # Create VAE or embedding token
        if self.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, attention_mask.to(torch.bool)), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)
            xseq = torch.cat((emb_token[None], x), 0)

            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, attention_mask), 1)

        # Add positional encoding and pass through Transformer encoder
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            return torch.distributions.Normal(mu, std)
        else:
            return final[0]
