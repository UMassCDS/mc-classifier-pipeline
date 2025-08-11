# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

You should also add project tags for each release in Github, see [Managing releases in a repository](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

## [Unreleased]

### Added
- Add `doc_retriever` script for fetching articles from Media Cloud.
- Add `label_studio_uploader` script for uploading data to Label Studio.
- Add `run_pipeline` script for connecting document-retrieval and label-studio-upload steps
- Add `preprocessing` script to fetch annotated tasks from Label Studio
- Add `bert_binary_recipe.py` for BERT-based text classification (training and inference) with HuggingFace Transformers.
- Add `evaluation.py` for evaluating multiple models and generating metrics summary with leaderboard
- Add `trainer.py` for training multiple model recipes from configuration files
- Add `sk_naive_bayes_recipe.py` for scikit-learn based text classification
- Add `inference.py` for generating predictions for a list of story URLs using a trained model
- Add `query_keyword_expander` script to generate a search query that can cast a wider net and retrieve more relevant stories from MC (optional)

### Changed
- Modify `doc_retriever` script to store in Label Studio formatted Json.
- Refactor parsing arguments in both `doc_retriever` and `label_studio_uploader`
- Modify `doc_retriever` script to accept optional collection ID



