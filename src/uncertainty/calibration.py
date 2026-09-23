"""
Confidence calibration and low-confidence gating (spec section 24;
reports/architecture_decision.md section 8).

Motivation, tied directly to a measured finding: the dataset audit
(dataset_feasibility_audit.md section 10) found basophils to be the smallest
class in every source dataset, and the real leakage/QC report
(datasets/processed/classification_v1/leakage_and_qc_report.json) confirms
this in the actual pooled training set (848 basophil vs. 11,220 rbc in train).
Minority classes trained on few examples are known to produce overconfident,
poorly-calibrated softmax outputs -- so calibration here is a direct response
to a measured problem, not generic polish.

Two pieces:
  1. TemperatureScaler: a single learned scalar T > 0 that divides the
     pre-softmax logits before the final softmax, fit on the validation set
     by minimizing negative log-likelihood (Guo et al. 2017 "On Calibration
     of Modern Neural Networks" -- a standard, low-risk, single-parameter
     method; no architecture change, cannot change which class is predicted,
     only how confident the reported probability is).
  2. classify_with_confidence_gate: applies a threshold to the (calibrated)
     top-1 probability and returns an explicit LOW_CONFIDENCE status instead
     of forcing a label -- spec section 24/9's "the system should be capable
     of saying LOW CONFIDENCE" requirement.
"""
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # temperature = exp(0) = 1.0 initially

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_temperature)

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 200):
        """val_logits: (N, num_classes) raw model outputs on a held-out val set.
        Returns the fitted temperature value (also stored in self)."""
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature


@dataclass
class ConfidenceGatedPrediction:
    status: str                       # "CONFIDENT" or "LOW_CONFIDENCE"
    predicted_class: Optional[str]    # None if LOW_CONFIDENCE
    top1_probability: float
    calibrated_probabilities: dict    # {class_name: probability}, always populated
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIDENCE_THRESHOLD = 0.55  # documented, tunable; not a claim of clinical significance


def classify_with_confidence_gate(
    logits: torch.Tensor,
    class_names: list,
    temperature: float = 1.0,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ConfidenceGatedPrediction:
    """logits: (num_classes,) raw model output for a single image."""
    if temperature <= 0:
        # Dividing logits by <= 0 yields inf/nan or an inverted distribution --
        # both silently corrupt every downstream confidence number.
        raise ValueError(f"temperature must be > 0, got {temperature!r}")
    calibrated_logits = logits / temperature
    probs = F.softmax(calibrated_logits, dim=0)
    top1_prob, top1_idx = probs.max(dim=0), probs.argmax(dim=0)
    top1_prob = float(probs[top1_idx].item())

    prob_dict = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

    if top1_prob < threshold:
        return ConfidenceGatedPrediction(
            status="LOW_CONFIDENCE",
            predicted_class=None,
            top1_probability=top1_prob,
            calibrated_probabilities=prob_dict,
            message=(
                "The morphology or image quality is ambiguous: no class reached the "
                f"{threshold:.0%} confidence threshold. Consider a sharper image, better "
                "illumination, or manual expert review."
            ),
        )

    return ConfidenceGatedPrediction(
        status="CONFIDENT",
        predicted_class=class_names[int(top1_idx.item())],
        top1_probability=top1_prob,
        calibrated_probabilities=prob_dict,
    )
