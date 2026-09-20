"""ClusterFewshot teleprompter for semantic-aware few-shot selection."""

from .cluster_fewshot import ClusterFewshot
from .semantic_encoder import (
    SemanticEncoder,
    create_hosted_encoder,
    create_numeric_encoder,
    create_sentence_transformer_encoder,
    default_text_extractor,
    numeric_extractor,
)

__all__ = [
    "ClusterFewshot",
    "SemanticEncoder",
    "default_text_extractor",
    "numeric_extractor",
    "create_sentence_transformer_encoder",
    "create_hosted_encoder",
    "create_numeric_encoder",
]
