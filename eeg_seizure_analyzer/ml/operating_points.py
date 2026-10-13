"""Recommended operating points for trained seizure models.

The UI fills the detection threshold, boundary threshold, convulsive classifier and
convulsive threshold from here when a model is chosen.

Sources, in order of preference:

1. ``metadata.json`` → ``"operating_point"``. A model can carry its own.
2. ``VALIDATED`` below: operating points chosen and validated outside training, for
   the models in use.
3. The training optimum from ``best_metrics``: the threshold that maximised event-F1
   on validation windows. This is not the same as a validated cascade operating
   point. The production model's training optimum is 0.7, but it runs at 0.5. Training
   records no boundary threshold, so that one is left as it is.
"""

from __future__ import annotations

import json

# threshold / boundary: Stage 1 (U-Net) core and hysteresis thresholds.
# convulsive_model / convulsive_threshold: the Stage-2 classifier it was validated with.
VALIDATED: dict[str, dict] = {
    "UNetv2_20260615": {
        "threshold": 0.5, "boundary": 0.1,
        "convulsive_model": "Convulsive_v4LUNARC_20260616", "convulsive_threshold": 0.45,
        "source": "SV2A production pipeline",
    },
    # RAM-GDNF retrained models: pre-registered rule, maximum cascade convulsive F1 on
    # the held-in batches over a joint Stage-1 x Stage-2 grid (NED-Net methods paper).
    "ramgdnf_temporal_pr": {
        "threshold": 0.95, "boundary": 0.5,
        "convulsive_model": "conv_temporal", "convulsive_threshold": 0.6,
        "source": "methods paper, pre-registered max-F1 selection",
    },
    "ramgdnf_all_prod_pr": {
        "threshold": 0.95, "boundary": 0.5,
        "convulsive_model": "conv_temporal", "convulsive_threshold": 0.6,
        "source": "methods paper, pre-registered max-F1 selection",
    },
    "ramgdnf_armA3_holdB3_pr": {
        "threshold": 0.9, "boundary": 0.5,
        "convulsive_model": "conv_armA_holdB3", "convulsive_threshold": 0.3,
        "source": "methods paper, pre-registered max-F1 selection",
    },
}


def _metadata(model_name: str) -> dict:
    from eeg_seizure_analyzer.ml.train import MODELS_DIR
    try:
        with open(MODELS_DIR / model_name / "metadata.json") as fh:
            return json.load(fh)
    except Exception:
        return {}


def classifier_threshold(conv_model: str) -> float | None:
    """A Stage-2 classifier's own training-optimum threshold."""
    bm = _metadata(conv_model).get("best_metrics") or {}
    t = bm.get("best_threshold", _metadata(conv_model).get("best_threshold"))
    return float(t) if t is not None else None


def recommended_operating_point(model_name: str) -> dict:
    """The operating point to pre-fill for a seizure model.

    Keys, each present only when known:
    - ``threshold``, ``boundary``;
    - ``convulsive_model`` (a Stage-2 classifier name, or "" for the U-Net's own
      convulsive channel);
    - ``convulsive_threshold``;
    - ``source`` (always present), for display.
    """
    if not model_name:
        return {"source": ""}
    meta = _metadata(model_name)
    op = meta.get("operating_point")
    if isinstance(op, dict) and op:
        return {**op, "source": op.get("source", "model metadata")}
    if model_name in VALIDATED:
        return dict(VALIDATED[model_name])
    bm = meta.get("best_metrics") or {}
    out: dict = {"source": "training optimum (validation event F1)"}
    if bm.get("best_threshold") is not None:
        out["threshold"] = float(bm["best_threshold"])
    if bm.get("conv_best_threshold") is not None:
        out["convulsive_threshold"] = float(bm["conv_best_threshold"])
    return out


def convulsive_threshold_for(model_name: str, conv_model: str) -> tuple[float | None, str]:
    """Convulsive threshold for a (U-Net, classifier) pair, and where it came from.

    The validated pairing's threshold if this is it, else the classifier's own
    training optimum. With no classifier, the U-Net channel's optimum.
    """
    op = recommended_operating_point(model_name)
    if conv_model == op.get("convulsive_model", None) and "convulsive_threshold" in op:
        return op["convulsive_threshold"], op["source"]
    if conv_model:
        return classifier_threshold(conv_model), f"classifier “{conv_model}” training optimum"
    bm = _metadata(model_name).get("best_metrics") or {}
    t = bm.get("conv_best_threshold")
    return (float(t) if t is not None else None), "U-Net convulsive channel, training optimum"
