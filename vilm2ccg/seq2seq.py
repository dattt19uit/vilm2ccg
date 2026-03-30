"""
seq2seq.py – Shared model and tokeniser definitions for ViLM2CCG training.

Provides:
  - ``CharTokenizer`` : character-level tokeniser with vocabulary persisted as JSON
  - ``PositionalEncoding`` : sinusoidal PE module
  - ``CircuitSeq2Seq`` : small Transformer encoder-decoder (PyTorch ``nn.Transformer``)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


class CharTokenizer:
    """Character-level tokeniser with a vocabulary built from the corpus.

    Every distinct character in the training corpus becomes a token.  Special
    tokens are prepended.  This approach handles Vietnamese diacritics and the
    ASCII characters that appear in CircuitJSON without any external library.
    """

    def __init__(self) -> None:
        self.token2id: dict[str, int] = {}
        self.id2token: list[str] = []

    def build(self, texts: list[str]) -> None:
        """Build vocabulary from a list of strings."""
        chars: set[str] = set()
        for text in texts:
            chars.update(text)
        vocab = SPECIAL_TOKENS + sorted(chars - set(SPECIAL_TOKENS))
        self.id2token = vocab
        self.token2id = {c: i for i, c in enumerate(vocab)}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> list[int]:
        unk = self.token2id[UNK_TOKEN]
        ids = [self.token2id.get(c, unk) for c in text]
        if add_bos:
            ids = [self.token2id[BOS_TOKEN]] + ids
        if add_eos:
            ids = ids + [self.token2id[EOS_TOKEN]]
        return ids

    def decode(self, ids: list[int]) -> str:
        bos = self.token2id.get(BOS_TOKEN, -1)
        eos = self.token2id.get(EOS_TOKEN, -1)
        pad = self.token2id.get(PAD_TOKEN, -1)
        chars: list[str] = []
        for i in ids:
            if i in (bos, pad):
                continue
            if i == eos:
                break
            chars.append(self.id2token[i] if i < len(self.id2token) else UNK_TOKEN)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token2id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token2id[EOS_TOKEN]

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id2token": self.id2token, "token2id": self.token2id}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.id2token = data["id2token"]
        tok.token2id = data["token2id"]
        return tok


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Seq2Seq model
# ---------------------------------------------------------------------------


class CircuitSeq2Seq(nn.Module):
    """Small Transformer encoder-decoder for NL → CircuitJSON generation.

    Default configuration (≈3.8 M parameters, CPU-friendly):
      d_model=256, nhead=8, 2 encoder + 2 decoder layers, FFN=1024.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt_len = tgt.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src.device)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_proj(out)  # (B, T, vocab_size)
