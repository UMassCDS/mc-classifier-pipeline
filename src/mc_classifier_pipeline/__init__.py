"""MC Classifier Pipeline package."""

__version__ = "0.0.1"

from . import (
    bert_recipe,
    doc_retriever,
    evaluation,
    label_studio_uploader,
    preprocessing,
    run_pipeline,
    sk_naive_bayes_recipe,
    trainer,
    utils,
)

__all__ = [
    "bert_recipe",
    "doc_retriever",
    "evaluation",
    "label_studio_uploader",
    "preprocessing",
    "run_pipeline",
    "sk_naive_bayes_recipe",
    "trainer",
    "utils",
    "__version__",
]
