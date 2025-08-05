# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

You should also add project tags for each release in Github, see [Managing releases in a repository](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

## [Unreleased]

- Add `doc_retriever` script for fetching articles from Media Cloud.
- Modify `doc_retriever` script to store in Label Studio formatted Json.
- Add `label_studio_uploader` script for uploading data to Label Studio.
- Refactor parsing arguments in both `doc_retriever` and `label_studio_uploader`
- Add `run_pipeline` script for connecting document-retrieval and label-studio-upload steps
- Modify `doc_retriever` script to accept optional collection ID
- Add `preprocessing` script to fetch annotated tasks from Label Studio
- Add `bert_binary_recipe.py` for BERT-based text classification (training and inference) with HuggingFace Transformers.
- Add `evaluation.py` for evaluating multiple models and generating metrics summary with leaderboard
- Add `trainer.py` for training multiple model recipes from configuration files

