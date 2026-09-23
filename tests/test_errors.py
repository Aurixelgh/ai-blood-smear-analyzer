"""Tests for src/errors.py -- the typed error taxonomy (engineering spec
Phase 6). These are pure-software contract tests; no model required."""
from src.errors import (
    AnalysisError, DataQualityError, ErrorCategory, InferenceError,
    ModelArtifactError, ModelNotAvailableError, UserError,
)


def test_every_error_has_a_category_and_message():
    for exc_type in (UserError, DataQualityError, ModelNotAvailableError,
                     ModelArtifactError, InferenceError):
        err = exc_type("some message")
        assert isinstance(err, AnalysisError)
        assert isinstance(err.category, ErrorCategory)
        assert str(err) == "some message"


def test_categories_are_distinctly_classified():
    assert UserError("x").category == ErrorCategory.USER_ERROR
    assert DataQualityError("x").category == ErrorCategory.DATA_QUALITY
    assert ModelNotAvailableError("x").category == ErrorCategory.MODEL_STATE
    assert ModelArtifactError("x").category == ErrorCategory.MODEL_STATE
    assert InferenceError("x").category == ErrorCategory.INFERENCE_FAILURE


def test_errors_are_runtime_errors_for_backward_compatibility():
    assert issubclass(AnalysisError, RuntimeError)


def test_details_are_preserved_for_logs():
    err = ModelArtifactError("weights corrupt", details="RuntimeError: boom")
    assert err.message == "weights corrupt"
    assert "boom" in err.details


def test_missing_model_is_model_state_not_inference_failure():
    """A pending model must never be reported as an inference failure --
    that distinction is the whole point of the taxonomy."""
    err = ModelNotAvailableError("not trained yet")
    assert err.category == ErrorCategory.MODEL_STATE
    assert err.category != ErrorCategory.INFERENCE_FAILURE
