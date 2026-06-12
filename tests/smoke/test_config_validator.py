"""Tests for the context-aware validator dispatch in ConfigurationValidator.

Pins a real bug found during the 2026-06-12 mypy cleanup: ``validate()``
dispatched any validator with ``co_argcount > 1`` to
``rule.validate_with_context(value, config)``, but that method was never
implemented -- registering any 2-arg validator made validation raise
AttributeError instead of producing a result.
"""

from giflab.config_validator import ConfigurationValidator, ValidationRule


def _make_validator() -> ConfigurationValidator:
    """A ConfigurationValidator with only our test rules registered."""
    validator = ConfigurationValidator()
    validator.rules = {}
    return validator


class TestValidateWithContext:
    """2-arg (value, config) validators must work through validate()."""

    def test_two_arg_validator_passing(self):
        validator = _make_validator()
        validator.add_rule(
            "FRAME_CACHE.memory_limit_mb",
            ValidationRule(
                "context",
                lambda value, config: value
                <= config.get("LIMITS", {}).get("max_mb", 0),
                "memory_limit_mb exceeds LIMITS.max_mb",
            ),
        )

        results = validator.validate(
            {
                "FRAME_CACHE": {"memory_limit_mb": 100},
                "LIMITS": {"max_mb": 500},
            }
        )

        assert results["error"] == []
        assert results["warning"] == []

    def test_two_arg_validator_failing(self):
        validator = _make_validator()
        validator.add_rule(
            "FRAME_CACHE.memory_limit_mb",
            ValidationRule(
                "context",
                lambda value, config: value
                <= config.get("LIMITS", {}).get("max_mb", 0),
                "memory_limit_mb exceeds LIMITS.max_mb",
            ),
        )

        results = validator.validate(
            {
                "FRAME_CACHE": {"memory_limit_mb": 1000},
                "LIMITS": {"max_mb": 500},
            }
        )

        # Same (is_valid, message) shape as 1-arg rules: the error message
        # lands under the rule's severity, prefixed with the path.
        assert results["error"] == [
            "FRAME_CACHE.memory_limit_mb: memory_limit_mb exceeds LIMITS.max_mb"
        ]

    def test_two_arg_validator_exception_is_wrapped(self):
        validator = _make_validator()

        def exploding(value, config):
            raise RuntimeError("boom")

        validator.add_rule(
            "FRAME_CACHE.memory_limit_mb",
            ValidationRule("context", exploding, "context check failed"),
        )

        results = validator.validate({"FRAME_CACHE": {"memory_limit_mb": 1}})

        # validate_with_context must wrap exceptions exactly like validate().
        assert len(results["error"]) == 1
        assert results["error"][0].startswith(
            "FRAME_CACHE.memory_limit_mb: context check failed"
        )
        assert "boom" in results["error"][0]

    def test_one_arg_rules_unaffected(self):
        validator = _make_validator()
        validator.add_rule(
            "FRAME_CACHE.memory_limit_mb",
            ValidationRule("positive", lambda value: value > 0, "must be positive"),
        )

        ok = validator.validate({"FRAME_CACHE": {"memory_limit_mb": 10}})
        bad = validator.validate({"FRAME_CACHE": {"memory_limit_mb": -1}})

        assert ok["error"] == []
        assert bad["error"] == ["FRAME_CACHE.memory_limit_mb: must be positive"]


class TestValidationRuleDirect:
    """ValidationRule.validate_with_context mirrors validate()'s contract."""

    def test_returns_true_none_on_pass(self):
        rule = ValidationRule("ctx", lambda v, c: True, "never")
        assert rule.validate_with_context(1, {}) == (True, None)

    def test_returns_false_message_on_fail(self):
        rule = ValidationRule("ctx", lambda v, c: False, "nope")
        assert rule.validate_with_context(1, {}) == (False, "nope")
