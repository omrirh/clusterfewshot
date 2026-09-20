# ClusterFewshot

Semantic-clustering-based few-shot demonstration selection for [DSPy](https://github.com/stanfordnlp/dspy).

Most few-shot optimizers pick demonstrations by random search or metric-based ranking, overlooking the semantic structure of the task. ClusterFewshot clusters training and validation examples in a shared embedding space, scores each candidate demonstration by its empirical effect as a one-shot example, then selects the demonstration set that performs best on the validation set. This substantially reduces optimization cost while consistently improving accuracy relative to prior bootstrap-based methods, in both standalone prompt tuning and hybrid prompt-weight optimization (see the paper below).

## Install

```bash
pip install -e .
```

Requires Python >= 3.10 and depends on `dspy`, `scikit-learn`, `sentence-transformers`, and `datasets`.

## Quickstart

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from clusterfewshot import ClusterFewshot, create_sentence_transformer_encoder

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predict(question=question)


dataset = GSM8K()

optimizer = ClusterFewshot(
    task_type="arithmetic",
    metric=gsm8k_metric,
    semantic_encoders=[create_sentence_transformer_encoder("all-mpnet-base-v2")],
)

optimized = optimizer.compile(student=CoT(), trainset=dataset.train, valset=dataset.dev)
```

Runnable versions: [`examples/quickstart_gsm8k.py`](examples/quickstart_gsm8k.py) (hosted API model) and [`examples/quickstart_gsm8k_local.py`](examples/quickstart_gsm8k_local.py) (any OpenAI-compatible local server, e.g. SGLang or vLLM).

## How it works

1. Clusters training and validation examples in a shared semantic embedding space (K selected via silhouette score).
2. Ranks each training example by its empirical effect as a one-shot demonstration, evaluated against a compact cluster-representative subset of the validation set.
3. Assembles a few-shot set from a few candidate sampling strategies (e.g. globally top-ranked vs. best-per-cluster) and keeps whichever performs best on the validation set.

## Bring-Your-Own-Encoder (BYOE)

`ClusterFewshot` takes one or more `SemanticEncoder` instances and grid-searches over them, keeping whichever produces the best clustering (highest silhouette score). A `SemanticEncoder` wraps two things:

- `semantics_extract`: picks the signal to embed out of a single `dspy.Example` (e.g. `ex.question`, or a numeric feature vector). Extraction only, no embedding happens here.
- `encoder`: turns the extracted signal into a vector, a local model, a `dspy.Embedder` instance, or `None` for numeric passthrough.

```python
from clusterfewshot import (
    create_sentence_transformer_encoder,  # local SentenceTransformer model
    create_hosted_encoder,                # hosted litellm embedding model, e.g. "openai/text-embedding-3-small"
    create_numeric_encoder,               # identity encoder for already-numeric inputs (e.g. Iris)
)

# Default extraction joins all input field values as text
encoder = create_sentence_transformer_encoder("all-mpnet-base-v2")

# Custom extraction for tasks with more than one relevant input field
encoder = create_sentence_transformer_encoder(
    "all-mpnet-base-v2",
    semantics_extract=lambda ex: f"{ex.question} {ex.some_other_field}",
)

# No local model download required
encoder = create_hosted_encoder("openai/text-embedding-3-small")
```

## When to use it

- vs. **BootstrapFewShotWithRandomSearch**: structured demonstration selection instead of random search.
- vs. **MIPROv2**: demonstration selection only, with a fixed and predictable call budget, instead of a joint instruction + demonstration search.
- vs. **GEPA**: complementary, GEPA optimizes instructions with no few-shot examples; ClusterFewshot is the cheap way to do the demonstration half.

## Citation

```bibtex
@article{barhaim2026clusterfewshot,
  title   = {ClusterFewshot: Improving Few-shot Optimization for LLMs workflow},
  author  = {Bar Haim, Omri and Katz, Shahar and Wolf, Lior},
  year    = {2026},
  note    = {Preprint}
}
```

Paper link: TBD (arxiv link pending).
