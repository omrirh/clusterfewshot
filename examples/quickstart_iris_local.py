"""Quickstart: ClusterFewshot on Iris classification with a local OpenAI-compatible model
(e.g. SGLang or vLLM serving Qwen3.5-9B on an A100).

No dataset download needed, Iris ships with scikit-learn. Useful as a sanity check
that sidesteps any issue with the GSM8K loader's HF Hub dependency.

Configure via env vars, matching whatever you deployed:
    export LOCAL_LM_MODEL=Qwen/Qwen3.5-9B      # must match the served model name
    export LOCAL_LM_API_BASE=http://localhost:7501/v1
    export LOCAL_LM_API_KEY=EMPTY              # most local servers don't check this

Run:
    python examples/quickstart_iris_local.py
"""

import os
import random

import dspy
from sklearn.datasets import load_iris

from clusterfewshot import ClusterFewshot, create_numeric_encoder

model = os.environ.get("LOCAL_LM_MODEL", "Qwen/Qwen3.5-9B")
api_base = os.environ.get("LOCAL_LM_API_BASE", "http://localhost:30000/v1")
api_key = os.environ.get("LOCAL_LM_API_KEY", "EMPTY")

lm = dspy.LM(f"openai/{model}", api_base=api_base, api_key=api_key)
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


class IrisClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("sepal_length, sepal_width, petal_length, petal_width -> species")

    def forward(self, sepal_length, sepal_width, petal_length, petal_width):
        return self.predict(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )


def iris_metric(example, prediction, trace=None):
    return example.species == prediction.species


iris = load_iris()
examples = [
    dspy.Example(
        sepal_length=float(row[0]),
        sepal_width=float(row[1]),
        petal_length=float(row[2]),
        petal_width=float(row[3]),
        species=iris.target_names[label],
    ).with_inputs("sepal_length", "sepal_width", "petal_length", "petal_width")
    for row, label in zip(iris.data, iris.target, strict=True)
]

# Iris is ordered by class (50 setosa, 50 versicolor, 50 virginica) - shuffle
# before slicing so every split actually contains all three species.
random.Random(0).shuffle(examples)

trainset = examples[:60]
valset = examples[60:90]
testset = examples[90:110]

optimizer = ClusterFewshot(
    task_type="classification",
    metric=iris_metric,
    semantic_encoders=[create_numeric_encoder()],
)

optimized = optimizer.compile(student=IrisClassifier(), trainset=trainset, valset=valset)

evaluate = dspy.Evaluate(devset=testset, metric=iris_metric, display_progress=True)
baseline_score = evaluate(IrisClassifier()).score
optimized_score = evaluate(optimized).score

num_demos = sum(len(predictor.demos) for _, predictor in optimized.named_predictors())
print(f"\nModel: {model} @ {api_base}")
print(f"Demos selected: {num_demos}")
print(f"Baseline (no demos) test accuracy:  {baseline_score:.1f}%")
print(f"ClusterFewshot test accuracy:       {optimized_score:.1f}%")
