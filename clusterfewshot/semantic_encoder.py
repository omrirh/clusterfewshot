"""
SemanticEncoder: Modular encoder interface for ClusterFewshot's Bring-Your-Own-Encoder design.

A SemanticEncoder wraps two things:

1. `semantics_extract`: picks the signal to embed out of a single dspy.Example
   (e.g. `ex.question`, or a numeric feature vector). Extraction only, no
   embedding happens here.
2. `encoder`: turns the extracted signal into a vector. Can be an object exposing
   `.encode(list) -> array` (e.g. SentenceTransformer), a `dspy.Embedder` instance
   (hosted litellm models, or any callable, with batching and caching built in),
   a plain callable, or None for numeric passthrough.
"""

import logging
from typing import Any, Callable

import numpy as np

import dspy
from dspy.primitives import Example

logger = logging.getLogger(__name__)


def default_text_extractor(example: Example) -> str:
    """
    Generic text extraction: joins all input field values as space-separated text.

    Reduces to just that field's value for single-input-field tasks (e.g. GSM8K's
    `question`). For multi-field tasks, pass a task-specific `semantics_extract`
    instead of relying on this default.
    """
    return " ".join(str(v) for v in example.inputs().values())


def numeric_extractor(example: Example) -> np.ndarray:
    """Generic numeric extraction: casts input field values to float and stacks them into a vector."""
    return np.array([float(v) for v in example.inputs().values()])


class SemanticEncoder:
    """
    Encapsulates an embedding backend and the Example-to-signal extraction it needs.

    Example usage:
        # Local SentenceTransformer model, default extraction (all input fields as text)
        encoder = create_sentence_transformer_encoder("all-mpnet-base-v2")

        # Custom extraction for a hypothetical multi-field task (whatever fields
        # your own Example schema actually has, e.g. a question plus some other field)
        encoder = create_sentence_transformer_encoder(
            "all-mpnet-base-v2",
            semantics_extract=lambda ex: f"{ex.question} {ex.some_other_field}",
        )

        # Hosted embedding model via litellm, no local model download
        encoder = create_hosted_encoder("openai/text-embedding-3-small")

        # Numeric/identity encoder for already-numeric inputs (e.g. Iris)
        encoder = create_numeric_encoder()
    """

    def __init__(
        self,
        encoder: Any,
        semantics_extract: Callable[[Example], Any] = default_text_extractor,
        name: str | None = None,
    ):
        """
        Initialize a SemanticEncoder.

        Args:
            encoder: The embedding backend. One of: an object exposing
                `.encode(list) -> array` (e.g. SentenceTransformer), a `dspy.Embedder`
                instance or any other callable `fn(list) -> array`, or None for a
                numeric passthrough (semantics_extract's output is used directly).
            semantics_extract: Picks the signal to embed out of a single Example,
                pure extraction independent of `encoder`. Defaults to joining all
                input field values as text.
            name: Optional name for the encoder (used for logging and identification).
        """
        self.encoder = encoder
        self.semantics_extract = semantics_extract
        self._name = name or self._infer_name()

    def _infer_name(self) -> str:
        """Infer encoder name from the encoder object if not explicitly provided."""
        if self.encoder is None:
            return "NumericEncoder"
        if hasattr(self.encoder, "model"):
            return str(self.encoder.model)
        if hasattr(self.encoder, "model_name"):
            return str(self.encoder.model_name)
        return type(self.encoder).__name__

    def encode(self, examples: list[Example]) -> np.ndarray:
        """
        Encode a list of examples into latent embedding vectors.

        Args:
            examples: List of Example objects to encode

        Returns:
            numpy array of shape (n_examples, embedding_dim) containing the embeddings
        """
        try:
            inputs = [self.semantics_extract(ex) for ex in examples]

            if self.encoder is None:
                embeddings = np.stack(inputs)
            elif hasattr(self.encoder, "encode"):
                embeddings = self.encoder.encode(inputs, convert_to_numpy=True)
            elif callable(self.encoder):
                embeddings = np.asarray(self.encoder(inputs))
            else:
                raise TypeError(
                    f"encoder must be None, expose .encode(), or be callable; got {type(self.encoder)}"
                )

            logger.debug(f"[{self._name}] Encoded {len(examples)} examples → shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"[{self._name}] Encoding failed: {e}")
            raise

    def name(self) -> str:
        """Return the encoder name."""
        return self._name

    def __str__(self) -> str:
        """String representation showing the encoder name."""
        return self._name

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"SemanticEncoder(name='{self._name}', encoder={type(self.encoder).__name__})"


# ============================================================================
# ENCODER FACTORY HELPERS
# ============================================================================

def create_sentence_transformer_encoder(
    model_name: str,
    semantics_extract: Callable[[Example], Any] = default_text_extractor,
    device: str = "cpu",
) -> SemanticEncoder:
    """
    Factory function to create a SemanticEncoder from a local SentenceTransformer model.

    Args:
        model_name: Name of the SentenceTransformer model (e.g., "all-mpnet-base-v2")
        semantics_extract: Picks the text to embed out of an Example. Defaults to
            joining all input field values; override for multi-field extraction.
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        SemanticEncoder configured with the SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embedder = dspy.Embedder(model.encode)
    return SemanticEncoder(encoder=embedder, semantics_extract=semantics_extract, name=model_name)


def create_hosted_encoder(
    model: str,
    semantics_extract: Callable[[Example], Any] = default_text_extractor,
    **embedder_kwargs: Any,
) -> SemanticEncoder:
    """
    Factory function to create a SemanticEncoder from a hosted embedding model via litellm.

    No local model download required. Any embedding model litellm supports works here
    (e.g. "openai/text-embedding-3-small").

    Args:
        model: litellm-style model string, e.g. "openai/text-embedding-3-small"
        semantics_extract: Picks the text to embed out of an Example. Defaults to
            joining all input field values; override for multi-field extraction.
        **embedder_kwargs: Extra kwargs forwarded to dspy.Embedder (e.g. batch_size, caching)

    Returns:
        SemanticEncoder configured with the hosted embedding model
    """
    embedder = dspy.Embedder(model, **embedder_kwargs)
    return SemanticEncoder(encoder=embedder, semantics_extract=semantics_extract, name=model)


def create_numeric_encoder() -> SemanticEncoder:
    """
    Factory function to create a numeric/identity encoder for classification tasks.

    Returns:
        SemanticEncoder that uses input features directly as embeddings
    """
    return SemanticEncoder(encoder=None, semantics_extract=numeric_extractor, name="NumericEncoder")
