from typing import TYPE_CHECKING, override

from torch import nn

from taxonomy_classifier.model.encoders import CnnConfig, CnnEncoder, TransformerEncoder
from taxonomy_classifier.model.heads import HeadConfig, HierarchicalHeads

if TYPE_CHECKING:
    import torch

    from taxonomy_classifier.model.encoders import TransformerConfig
    from taxonomy_classifier.model.labels import LabelSpace

type EncoderConfig = CnnConfig | TransformerConfig


class TaxonomyClassifier(nn.Module):
    def __init__(self, encoder: CnnEncoder | TransformerEncoder, heads: HierarchicalHeads) -> None:
        super().__init__()

        self.encoder = encoder
        self.heads = heads

    @override
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = self.heads(self.encoder(tokens, mask))

        return logits


def build_classifier(
    encoder_config: EncoderConfig,
    label_space: LabelSpace,
    head_config: HeadConfig | None = None,
) -> TaxonomyClassifier:
    encoder: CnnEncoder | TransformerEncoder

    if isinstance(encoder_config, CnnConfig):
        encoder = CnnEncoder(encoder_config)

    else:
        encoder = TransformerEncoder(encoder_config)

    heads = HierarchicalHeads(encoder.output_dim, label_space.sizes, head_config or HeadConfig())

    return TaxonomyClassifier(encoder, heads)


def parameter_counts(model: TaxonomyClassifier) -> dict[str, int]:
    encoder = _count(model.encoder)
    heads = _count(model.heads)

    return {"encoder": encoder, "heads": heads, "total": encoder + heads}


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
