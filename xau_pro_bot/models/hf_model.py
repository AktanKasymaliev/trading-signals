"""Lazy Hugging Face trading model adapter."""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

_SKLEARN_FILENAMES = (
    "model.joblib",
    "model.pkl",
    "classifier.joblib",
    "trading_model.joblib",
)
_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _is_commit_sha(revision: str | None) -> bool:
    return bool(revision and _COMMIT_SHA_RE.fullmatch(revision))


class HFTradingModel:
    """Lazy-loaded optional model adapter for AI signal confirmation."""

    _LABEL_ALIASES = {
        "BUY": "BUY",
        "LONG": "BUY",
        "UP": "BUY",
        "SELL": "SELL",
        "SHORT": "SELL",
        "DOWN": "SELL",
        "NO_TRADE": "NO_TRADE",
        "NOTRADE": "NO_TRADE",
        "HOLD": "NO_TRADE",
        "NEUTRAL": "NO_TRADE",
        "FLAT": "NO_TRADE",
    }

    def __init__(
        self,
        model_id: str,
        model_type: str = "sklearn",
        cache_dir: str | None = None,
        revision: str | None = None,
        filename: str = "",
        local_path: str = "",
    ) -> None:
        self.model_id = model_id
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.revision = revision
        self.filename = filename
        self.local_path = local_path
        self._model: Any | None = None

    def _neutral(self, error: str) -> dict[str, Any]:
        return {
            "direction": "NO_TRADE",
            "confidence": 0.0,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
            "error": error,
        }

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_id and not self.local_path:
            raise RuntimeError("model_id is required for Hugging Face model loading")

        if self.model_type == "sklearn":
            self._model = self._load_sklearn()
        elif self.model_type == "transformers":
            self._model = self._load_transformers()
        elif self.model_type == "custom":
            raise NotImplementedError("custom Hugging Face model adapters are not implemented")
        else:
            raise RuntimeError(f"unsupported Hugging Face model_type: {self.model_type}")

        return self._model

    def _load_sklearn(self) -> Any:
        if self.local_path:
            log.info("Loading local sklearn artifact from %s", self.local_path)
            return joblib.load(self.local_path)
        if not _is_commit_sha(self.revision):
            raise RuntimeError(
                "sklearn/joblib models require a pinned 40-character commit SHA "
                "for Hugging Face artifacts"
            )

        if self.filename:
            candidates = (self.filename,)
        else:
            candidates = _SKLEARN_FILENAMES

        last_error: Exception | None = None
        for filename in candidates:
            try:
                log.info(
                    "Attempting to download Hugging Face sklearn artifact %s from %s",
                    filename,
                    self.model_id,
                )
                path = hf_hub_download(
                    repo_id=self.model_id,
                    filename=filename,
                    cache_dir=self.cache_dir,
                    revision=self.revision,
                )
                model = joblib.load(path)
                log.info(
                    "Loaded Hugging Face sklearn model %s from artifact %s",
                    self.model_id,
                    filename,
                )
                return model
            except Exception as exc:
                last_error = exc
                log.debug(
                    "Unable to load Hugging Face sklearn artifact %s from %s",
                    filename,
                    self.model_id,
                    exc_info=True,
                )

        if last_error is None:
            raise RuntimeError("no sklearn artifact filenames were configured")
        raise last_error

    def _load_transformers(self) -> Any:
        try:
            transformers = importlib.import_module("transformers")
            importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "optional transformers dependency is missing; install transformers and torch "
                "to use model_type='transformers'"
            ) from exc

        log.info("Creating Hugging Face transformers pipeline for %s", self.model_id)
        return transformers.pipeline(
            "text-classification",
            model=self.model_id,
            tokenizer=self.model_id,
        )

    def _normalize_class(self, value: Any) -> str | None:
        """Map a model class label to BUY/SELL/NO_TRADE, or None if unrecognized."""
        if value == 1:
            return "BUY"
        if value == -1:
            return "SELL"
        if value == 0:
            return "NO_TRADE"
        label = str(value).strip().upper().replace(" ", "_").replace("-", "_")
        return self._LABEL_ALIASES.get(label)

    def _predict_sklearn(self, model: Any, features: pd.DataFrame) -> dict[str, Any]:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features))[0]
            classes = getattr(model, "classes_", np.arange(len(probabilities)))
            mapped: dict[str, float] = {}
            for cls, probability in zip(classes, probabilities):
                normalized = self._normalize_class(cls)
                if normalized is None:
                    continue
                mapped[normalized] = mapped.get(normalized, 0.0) + float(probability)

            if not mapped:
                raise RuntimeError(
                    f"unrecognized model classes_: {list(classes)!r}; "
                    "expected BUY/SELL/NO_TRADE (or LONG/SHORT/FLAT) labels"
                )

            prob_buy = mapped.get("BUY")
            prob_sell = mapped.get("SELL")
            prob_no_trade = mapped.get("NO_TRADE")
            direction, confidence = max(mapped.items(), key=lambda item: item[1])
            return {
                "direction": direction,
                "confidence": float(confidence),
                "prob_buy": prob_buy,
                "prob_sell": prob_sell,
                "prob_no_trade": prob_no_trade,
            }

        prediction = np.asarray(model.predict(features))[0]
        normalized = self._normalize_class(prediction)
        if normalized is None:
            raise RuntimeError(
                f"unrecognized model.predict output: {prediction!r}"
            )
        return {
            "direction": normalized,
            "confidence": 0.50,
            "prob_buy": None,
            "prob_sell": None,
            "prob_no_trade": None,
        }

    def _predict_transformers(self, model: Any, features: pd.DataFrame) -> dict[str, Any]:
        raw = model(features.to_json(orient="records"))
        first = raw[0] if isinstance(raw, list) and raw else raw
        if not isinstance(first, dict):
            raise RuntimeError("transformers model returned an unsupported prediction format")

        direction = self._normalize_class(first.get("label", "NO_TRADE"))
        confidence = float(first.get("score", 0.0))
        return {
            "direction": direction,
            "confidence": confidence,
            "prob_buy": confidence if direction == "BUY" else None,
            "prob_sell": confidence if direction == "SELL" else None,
            "prob_no_trade": confidence if direction == "NO_TRADE" else None,
        }

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """Return a normalized trading prediction; failures become neutral."""
        try:
            model = self._load()
            if self.model_type == "sklearn":
                return self._predict_sklearn(model, features)
            if self.model_type == "transformers":
                return self._predict_transformers(model, features)
            raise NotImplementedError(
                "custom Hugging Face model adapters are not implemented"
            )
        except Exception as exc:
            log.exception("Hugging Face model prediction failed")
            return self._neutral(str(exc))
