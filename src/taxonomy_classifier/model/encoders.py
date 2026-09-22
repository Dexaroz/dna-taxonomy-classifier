from dataclasses import dataclass
import math
from typing import Final, override

import torch
from torch import nn
import torch.nn.functional as F

from taxonomy_classifier.model.tokens import PAD, VOCAB_SIZE

_ROPE_BASE: Final = 10_000.0


@dataclass(frozen=True, slots=True)
class CnnConfig:
    embedding_dim: int = 64
    channels: tuple[int, ...] = (128, 192, 256, 320)
    blocks_per_stage: int = 2
    kernel_size: int = 9
    dilations: tuple[int, ...] = (1, 2)
    output_dim: int = 256
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not self.channels or self.blocks_per_stage < 1 or self.kernel_size % 2 == 0:
            msg = "CnnConfig needs at least one stage, one block per stage and an odd kernel size"
            raise ValueError(msg)

        if len(self.dilations) != self.blocks_per_stage:
            msg = "CnnConfig needs one dilation per block in a stage"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    embedding_dim: int = 64
    model_dim: int = 256
    token_kernel: int = 8
    token_stride: int = 4
    layers: int = 6
    heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim % self.heads or (self.model_dim // self.heads) % 2:
            msg = "model_dim must split into heads of even dimension"
            raise ValueError(msg)

        if self.token_stride < 1 or self.token_kernel < self.token_stride or self.layers < 1:
            msg = "TransformerConfig needs token_kernel >= token_stride >= 1 and at least one layer"
            raise ValueError(msg)


class CnnEncoder(nn.Module):
    def __init__(self, config: CnnConfig) -> None:
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim, padding_idx=PAD)
        self.stem = nn.Conv1d(
            config.embedding_dim, config.channels[0], config.kernel_size, padding="same"
        )

        stages: list[nn.Module] = []
        previous = config.channels[0]

        for stage, width in enumerate(config.channels):
            if stage:
                stages.append(_Downsample(previous, width))

            stages.extend(
                _ResidualBlock(width, config.kernel_size, dilation, config.dropout)
                for dilation in config.dilations
            )

            previous = width

        self.stages = nn.ModuleList(stages)
        self.projection = nn.Linear(2 * previous, config.output_dim)
        self.norm = nn.LayerNorm(config.output_dim)
        self.output_dim = config.output_dim

    @override
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(1)
        hidden = self.stem(self.embedding(tokens).transpose(1, 2)) * valid

        for stage in self.stages:
            if isinstance(stage, _Downsample):
                hidden, valid = stage(hidden, valid)

                continue

            hidden = stage(hidden) * valid

        pooled = torch.cat([_masked_mean(hidden, valid), _masked_max(hidden, valid)], dim=-1)
        normalized: torch.Tensor = self.norm(self.projection(pooled))

        return normalized


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.kernel = config.token_kernel
        self.stride = config.token_stride

        self.embedding = nn.Embedding(VOCAB_SIZE, config.embedding_dim, padding_idx=PAD)
        self.tokenizer = nn.Conv1d(
            config.embedding_dim, config.model_dim, config.token_kernel, stride=config.token_stride
        )

        self.blocks = nn.ModuleList(_AttentionBlock(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.model_dim)
        self.output_dim = config.model_dim

    @override
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shortfall = self.kernel - tokens.shape[1]

        if shortfall > 0:
            tokens = F.pad(tokens, (0, shortfall), value=PAD)
            mask = F.pad(mask, (0, shortfall), value=False)

        embedded = self.embedding(tokens).transpose(1, 2) * mask.unsqueeze(1)
        hidden = self.tokenizer(embedded).transpose(1, 2)

        lengths = mask.sum(dim=-1).clamp(min=self.kernel)
        token_counts = (lengths - self.kernel) // self.stride + 1
        valid = torch.arange(hidden.shape[1], device=hidden.device) < token_counts.unsqueeze(-1)

        for block in self.blocks:
            hidden = block(hidden, valid)

        normalized: torch.Tensor = self.norm(
            _masked_mean(hidden.transpose(1, 2), valid.unsqueeze(1))
        )

        return normalized


class _ChannelNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(channels)

    @override
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized: torch.Tensor = self.norm(hidden.transpose(1, 2))

        return normalized.transpose(1, 2)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()

        self.norm = _ChannelNorm(channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding="same")
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    @override
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = self.pointwise(F.gelu(self.conv(F.gelu(self.norm(hidden)))))

        dropped: torch.Tensor = self.dropout(update)

        return hidden + dropped


class _Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    @override
    def forward(
        self, hidden: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        downsampled_valid = valid[..., ::2]

        return self.conv(hidden) * downsampled_valid, downsampled_valid


class _AttentionBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.dropout = config.dropout

        self.attention_norm = nn.LayerNorm(config.model_dim)
        self.qkv = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.output = nn.Linear(config.model_dim, config.model_dim)

        self.ffn_norm = nn.LayerNorm(config.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, config.ffn_multiplier * config.model_dim),
            nn.GELU(),
            nn.Linear(config.ffn_multiplier * config.model_dim, config.model_dim),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    @override
    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.residual_dropout(self._attend(self.attention_norm(hidden), valid))

        update: torch.Tensor = self.residual_dropout(self.ffn(self.ffn_norm(hidden)))

        return hidden + update

    def _attend(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape

        query, key, value = (
            self.qkv(hidden)
            .view(batch, length, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )

        query, key = rotate(query), rotate(key)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )

        projected: torch.Tensor = self.output(
            attended.transpose(1, 2).reshape(batch, length, width)
        )

        return projected


def rotate(states: torch.Tensor) -> torch.Tensor:
    length, dim = states.shape[-2], states.shape[-1]
    half = dim // 2

    frequencies = _ROPE_BASE ** (
        -torch.arange(half, device=states.device, dtype=torch.float32) / half
    )
    angles = (
        torch.arange(length, device=states.device, dtype=torch.float32)[:, None]
        * frequencies[None, :]
    )
    cos, sin = angles.cos().to(states.dtype), angles.sin().to(states.dtype)

    first, second = states[..., :half], states[..., half:]

    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


def _masked_mean(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(hidden.dtype)

    return (hidden * weights).sum(dim=-1) / weights.sum(dim=-1).clamp(min=1.0)


def _masked_max(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return hidden.masked_fill(~valid, -math.inf).amax(dim=-1)
