"""MC Classifier Pipeline package."""

__version__ = "0.0.1"

from mc_classifier_pipeline import (
    base_classifier,
    bert_recipe,
    doc_retriever,
    evaluation,
    label_studio_uploader,
    prediction,
    preprocessing,
    run_pipeline,
    sk_naive_bayes_recipe,
    trainer,
    utils,
)

__all__ = [
    "base_classifier",
    "bert_recipe",
    "doc_retriever",
    "evaluation",
    "label_studio_uploader",
    "prediction",
    "preprocessing",
    "run_pipeline",
    "sk_naive_bayes_recipe",
    "trainer",
    "utils",
    "__version__",
]
