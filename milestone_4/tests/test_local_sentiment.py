from pathlib import Path

import pytest

from src.local_sentiment import LABELS, SentimentEngine, SentimentModelError


def test_label_contract():
    assert LABELS == ("NEGATIVE", "NEUTRAL", "POSITIVE")


def test_missing_model_fails_clearly(tmp_path: Path):
    with pytest.raises(SentimentModelError):
        SentimentEngine(model_dir=tmp_path / "missing-model")
