# Reproducible ML Experiments

A lightweight, config-driven template for structured and reproducible machine learning experiments.

⚠️ Work in progress – minimal implementation in progress.

## Motivation

Reproducibility is fundamental to scientific machine learning.  
This repository provides a minimal yet disciplined experiment framework including:

- Config-driven pipelines
- Deterministic seed control
- Structured logging
- Cross-validation template
- Automated testing (CI)

The goal is to make experimental results traceable, comparable, and repeatable.

## Features
- Config-driven training (`configs/*.yaml`)
- Deterministic seed control
- K-fold cross-validation
- Structured logging to `results/runs.csv`
- Config snapshots saved to `results/config_<id>.json`

## Run
```bash
pip install -e .
python -m src.train --config configs/baseline.yaml
python -m src.train --config configs/sweep.yaml
```
## Required dependency note
Your `pyproject.toml` must include:
- `pyyaml`
- `numpy`
- `scikit-learn`

## Project Structure

configs/        # Experiment configurations
src/            # Core training and evaluation logic
results/        # Structured run outputs
notebooks/      # Result analysis
scripts/        # Execution helpers

## Reproducibility Protocol

Each run:

- Fixes random seeds across numpy / torch / python
- Saves full config snapshot
- Logs metrics to `results/runs.csv`
- Stores timestamp and experiment ID

Experiments can be reproduced by re-running:

python -m src.train --config configs/baseline.yaml

## Results

| Model | CV Accuracy | Seed |
|-------|------------|------|
| Logistic Regression | 0.842 | 42 |
| MLP | 0.867 | 42 |
