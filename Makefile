.PHONY: help setup data baseline bilstm transformer evaluate explain test lint format api dashboard all clean

PY ?= python

help:
	@echo "Bilingual Political Attitude Classification (English & Bangla)"
	@echo "Targets:"
	@echo "  setup       Install dependencies (pip install -r requirements.txt + package)"
	@echo "  data        Generate synthetic corpus and build train/val/test splits"
	@echo "  baseline    Train TF-IDF + Logistic Regression / SVM baseline"
	@echo "  bilstm      Train BiLSTM + attention model"
	@echo "  transformer Fine-tune multilingual transformer (DistilBERT/XLM-R)"
	@echo "  evaluate    Evaluate a trained model and emit metrics + figures"
	@echo "  explain     Run SHAP/LIME explainability on a trained model"
	@echo "  test        Run the unit-test suite"
	@echo "  lint        Run ruff + mypy"
	@echo "  format      Auto-format with black + ruff --fix"
	@echo "  api         Launch the FastAPI inference service"
	@echo "  dashboard   Launch the Streamlit dashboard"
	@echo "  all         data -> baseline -> bilstm -> transformer -> evaluate"

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

data:
	$(PY) -m political_sentiment.data.build_dataset --config configs/base.yaml

baseline:
	$(PY) -m political_sentiment.training.train --config configs/baseline_tfidf.yaml

bilstm:
	$(PY) -m political_sentiment.training.train --config configs/bilstm_attention.yaml

transformer:
	$(PY) -m political_sentiment.training.train --config configs/transformer_bert.yaml

evaluate:
	$(PY) -m political_sentiment.evaluation.evaluate --config $(CONFIG)

compare:
	$(PY) -m political_sentiment.training.compare --configs \
		configs/baseline_tfidf.yaml configs/baseline_svm.yaml \
		configs/bilstm_attention.yaml configs/transformer_bert.yaml

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src

format:
	$(PY) -m black src tests
	$(PY) -m ruff check --fix src tests

api:
	$(PY) -m uvicorn political_sentiment.api.app:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run src/political_sentiment/dashboard/app.py

all: data baseline bilstm transformer

clean:
	rm -rf artifacts/ data/processed/*.parquet data/synthetic/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} +
