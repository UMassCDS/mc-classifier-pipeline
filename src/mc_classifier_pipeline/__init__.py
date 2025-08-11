"""MC Classifier Pipeline package."""


__version__ = "0.0.1"

from mc_classifier_pipeline import (
    bert_recipe,
    doc_retriever,
    evaluation,
    label_studio_uploader,
    preprocessing,
    run_pipeline,
    sk_naive_bayes_recipe,
    trainer,
    utils,
    model_orchestrator,
    annotation_analysis
)

__all__ = [
    "annotation_analysis",
    "bert_recipe",
    "doc_retriever",
    "evaluation",
    "label_studio_uploader",
    "preprocessing",
    "run_pipeline",
    "sk_naive_bayes_recipe",
    "trainer",
    "utils",
    "model_orchestrator",
    "__version__",
]
