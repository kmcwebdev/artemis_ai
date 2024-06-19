# src/testing/io/huggingface_model_dataset.py

from pathlib import Path
from transformers import AutoModelForSequenceClassification
from kedro.io.core import AbstractDataSet, DataSetError

class HuggingFaceModelDataset(AbstractDataSet):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> AutoModelForSequenceClassification:
        try:
            return AutoModelForSequenceClassification.from_pretrained(self._filepath)
        except Exception as e:
            raise DataSetError(f"Failed to load model from {self._filepath}: {e}")

    def _save(self, model: AutoModelForSequenceClassification) -> None:
        try:
            model.save_pretrained(self._filepath)
        except Exception as e:
            raise DataSetError(f"Failed to save model to {self._filepath}: {e}")

    def _describe(self) -> dict:
        return dict(filepath=self._filepath)
