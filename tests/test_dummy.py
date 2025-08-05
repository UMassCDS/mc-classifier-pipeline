"""Basic tests for the MC Classifier Pipeline package."""

import pytest
from mc_classifier_pipeline import __version__


def test_package_version():
    """Test that the package has a version."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_package_imports():
    """Test that all main modules can be imported."""
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
    )

    # Test that modules exist
    assert bert_recipe is not None
    assert doc_retriever is not None
    assert evaluation is not None
    assert label_studio_uploader is not None
    assert preprocessing is not None
    assert run_pipeline is not None
    assert sk_naive_bayes_recipe is not None
    assert trainer is not None
    assert utils is not None


def test_utils_configure_logging():
    """Test that logging configuration works."""
    from mc_classifier_pipeline.utils import configure_logging

    # Should not raise an exception
    configure_logging()
