"""
Typed error taxonomy shared by the whole application (engineering spec
Phase 6: "Errors should distinguish between USER ERROR / DATA QUALITY /
MODEL STATE / INFERENCE FAILURE").

Design rules:
  1. Every error carries a `category` and a short, user-facing `message`.
     The UI layer renders `str(exc)` and may style by `category` -- it must
     never show a raw Python traceback to the user (spec section 33).
  2. Errors are raised, never swallowed: a failure must surface as a typed
     error or an explicit non-OK status, never as a generic success with
     placeholder data (the project's no-fabrication rule).
  3. `AnalysisError` subclasses `RuntimeError` on purpose: existing callers
     that catch broad exceptions keep working while this taxonomy is adopted
     incrementally.

Scientific limitations (a feature genuinely not computable for an input) are
NOT raised as errors -- the morphology engine reports them as `None` values
with an `unavailable_reasons` string, so a partial result still returns.
"""
from enum import Enum


class ErrorCategory(str, Enum):
    USER_ERROR = "USER_ERROR"                # bad input the user can fix themselves
    DATA_QUALITY = "DATA_QUALITY"            # input decodes fine but is unusable quality
    MODEL_STATE = "MODEL_STATE"              # model/weights missing, corrupt, or incompatible
    INFERENCE_FAILURE = "INFERENCE_FAILURE"  # a real internal failure during processing


class AnalysisError(RuntimeError):
    """Base class for every typed failure raised by this codebase."""

    category: ErrorCategory = ErrorCategory.INFERENCE_FAILURE

    def __init__(self, message: str, *, details: str = None):
        super().__init__(message)
        self.message = message
        self.details = details


class UserError(AnalysisError):
    """The user supplied something invalid (missing/unsupported/undecodable file)."""
    category = ErrorCategory.USER_ERROR


class DataQualityError(AnalysisError):
    """The image decoded fine but is too small/blurry/dark/bright to analyze."""
    category = ErrorCategory.DATA_QUALITY


class ModelNotAvailableError(AnalysisError):
    """Required model weights are not present yet (training not run or still in
    progress). This is a MODEL STATE, not a failure -- the app must show a
    clear pending state and must never fabricate a prediction instead."""
    category = ErrorCategory.MODEL_STATE


class ModelArtifactError(AnalysisError):
    """Model file exists but is corrupt, incomplete, or incompatible with the
    current code (e.g. missing checkpoint keys, unknown backbone)."""
    category = ErrorCategory.MODEL_STATE


class InferenceError(AnalysisError):
    """An unexpected failure occurred during a model forward pass or pipeline
    step. The real cause is preserved in `details` for logs -- never hidden
    behind a generic success message."""
    category = ErrorCategory.INFERENCE_FAILURE
