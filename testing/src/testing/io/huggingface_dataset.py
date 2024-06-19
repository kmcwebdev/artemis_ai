from typing import Any, Dict
from kedro.io import AbstractDataset
from datasets import load_from_disk

class HuggingFaceDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> Any:
        return load_from_disk(self._filepath)

    def _save(self, data: Any) -> None:
        data.save_to_disk(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)
