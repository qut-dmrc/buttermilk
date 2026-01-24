"""Unit tests for ImageRecord sad robot fallback behavior.

Tests verify that sad robot is only used for content policy refusals,
not for general API failures or technical errors.
"""

import pytest
from PIL import Image

from buttermilk._core.image import ImageRecord


class TestImageRecordRefusalDetection:
    """Test the refusal detection logic."""

    def test_is_refusal_error_with_none(self):
        """Test that None error is not considered a refusal."""
        assert not ImageRecord._is_refusal_error(None)

    def test_is_refusal_error_with_empty_string(self):
        """Test that empty string is not considered a refusal."""
        assert not ImageRecord._is_refusal_error("")

    def test_is_refusal_error_with_refusal_keyword(self):
        """Test detection of 'refusal' keyword."""
        assert ImageRecord._is_refusal_error("The request was refused")
        assert ImageRecord._is_refusal_error("Request Refusal: prompt violates policy")
        assert ImageRecord._is_refusal_error("Refusal")

    def test_is_refusal_error_with_content_policy_keyword(self):
        """Test detection of 'content policy' keyword."""
        assert ImageRecord._is_refusal_error("Content policy violation")
        assert ImageRecord._is_refusal_error("This violates the content policy")

    def test_is_refusal_error_with_blocked_keyword(self):
        """Test detection of 'blocked' keyword for policy blocks."""
        assert ImageRecord._is_refusal_error("Content blocked by safety filter")
        assert ImageRecord._is_refusal_error("Request blocked")

    def test_is_refusal_error_with_unsafe_keyword(self):
        """Test detection of 'unsafe' keyword."""
        assert ImageRecord._is_refusal_error("Image generation unsafe")
        assert ImageRecord._is_refusal_error("Prompt contains unsafe content")

    def test_is_refusal_error_with_not_allowed(self):
        """Test detection of 'not allowed' keyword."""
        assert ImageRecord._is_refusal_error("This is not allowed")
        assert ImageRecord._is_refusal_error("Not permitted by policy")

    def test_is_refusal_error_with_dict_error(self):
        """Test refusal detection with dict error format."""
        error_dict = {"message": "Content policy violation", "type": "policy_error"}
        assert ImageRecord._is_refusal_error(error_dict)

    def test_is_refusal_error_case_insensitive(self):
        """Test that refusal detection is case-insensitive."""
        assert ImageRecord._is_refusal_error("REFUSAL")
        assert ImageRecord._is_refusal_error("Refusal")
        assert ImageRecord._is_refusal_error("refusal")
        assert ImageRecord._is_refusal_error("CONTENT POLICY")

    def test_is_not_refusal_error_with_network_error(self):
        """Test that network errors are not considered refusals."""
        assert not ImageRecord._is_refusal_error("Connection timeout")
        assert not ImageRecord._is_refusal_error("Network error")
        assert not ImageRecord._is_refusal_error("HTTP 500 Internal Server Error")
        assert not ImageRecord._is_refusal_error("Failed to connect to API")

    def test_is_not_refusal_error_with_auth_error(self):
        """Test that authentication errors are not considered refusals."""
        assert not ImageRecord._is_refusal_error("Invalid API key")
        assert not ImageRecord._is_refusal_error("Authentication failed")
        assert not ImageRecord._is_refusal_error("Unauthorized")

    def test_is_not_refusal_error_with_rate_limit(self):
        """Test that rate limiting is not considered a refusal."""
        assert not ImageRecord._is_refusal_error("Rate limit exceeded")
        assert not ImageRecord._is_refusal_error("Too many requests")

    def test_is_not_refusal_error_with_parameter_error(self):
        """Test that invalid parameter errors are not considered refusals."""
        assert not ImageRecord._is_refusal_error("Invalid image size")
        assert not ImageRecord._is_refusal_error("Unknown parameter: foo")
        assert not ImageRecord._is_refusal_error("Invalid prompt format")


class TestImageRecordSadRobotFallback:
    """Test sad robot fallback behavior for ImageRecord."""

    def test_sad_robot_for_refusal_string_error(self):
        """Test that sad robot is used when image is None and error is a refusal (string)."""
        error_record = ImageRecord(
            image=None,
            error="The prompt violates content policy and was refused",
            prompt="test prompt",
            model="test-model",
        )

        # Should have sad robot image (not raise exception)
        assert error_record.image is not None
        assert isinstance(error_record.image, Image.Image)

    def test_sad_robot_for_refusal_dict_error(self):
        """Test that sad robot is used when image is None and error is a refusal (dict)."""
        error_record = ImageRecord(
            image=None,
            error={"message": "Content policy violation", "type": "safety_filter"},
            prompt="test prompt",
            model="test-model",
        )

        # Should have sad robot image (not raise exception)
        assert error_record.image is not None
        assert isinstance(error_record.image, Image.Image)

    def test_sad_robot_for_blocked_error(self):
        """Test sad robot for blocked content error."""
        error_record = ImageRecord(
            image=None,
            error="Image generation blocked due to safety filter",
            prompt="test prompt",
            model="test-model",
        )

        assert error_record.image is not None
        assert isinstance(error_record.image, Image.Image)

    def test_sad_robot_for_unsafe_error(self):
        """Test sad robot for unsafe content error."""
        error_record = ImageRecord(
            image=None,
            error="Request blocked: prompt contains unsafe content",
            prompt="test prompt",
            model="test-model",
        )

        assert error_record.image is not None
        assert isinstance(error_record.image, Image.Image)

    def test_no_sad_robot_for_network_error(self):
        """Test that network errors raise exception instead of using sad robot."""
        with pytest.raises(ValueError, match="technical error"):
            ImageRecord(
                image=None,
                error="Connection timeout while calling API",
                prompt="test prompt",
                model="test-model",
            )

    def test_no_sad_robot_for_auth_error(self):
        """Test that authentication errors raise exception instead of using sad robot."""
        with pytest.raises(ValueError, match="technical error"):
            ImageRecord(
                image=None,
                error="Invalid API key or credentials",
                prompt="test prompt",
                model="test-model",
            )

    def test_no_sad_robot_for_rate_limit_error(self):
        """Test that rate limit errors raise exception instead of using sad robot."""
        with pytest.raises(ValueError, match="technical error"):
            ImageRecord(
                image=None,
                error="Rate limit exceeded: too many requests",
                prompt="test prompt",
                model="test-model",
            )

    def test_no_sad_robot_for_http_error(self):
        """Test that HTTP errors raise exception instead of using sad robot."""
        with pytest.raises(ValueError, match="technical error"):
            ImageRecord(
                image=None,
                error="HTTP 500 Internal Server Error",
                prompt="test prompt",
                model="test-model",
            )

    def test_no_sad_robot_for_parsing_error(self):
        """Test that response parsing errors raise exception instead of using sad robot."""
        with pytest.raises(ValueError, match="technical error"):
            ImageRecord(
                image=None,
                error="Failed to parse API response: invalid JSON",
                prompt="test prompt",
                model="test-model",
            )

    def test_error_without_image_raises_exception(self):
        """Test that error without image requires error field to be set."""
        with pytest.raises(ValueError, match="Image is required"):
            ImageRecord(
                image=None,
                error=None,
                prompt="test prompt",
                model="test-model",
            )

    def test_error_field_converted_to_dict(self):
        """Test that string error is converted to dict format."""
        error_record = ImageRecord(
            image=None,
            error="Content policy violation",
            prompt="test prompt",
            model="test-model",
        )

        # Error should be converted to dict
        assert isinstance(error_record.error, dict)
        assert "message" in error_record.error
        assert error_record.error["message"] == "Content policy violation"

    def test_successful_generation_with_real_image(self):
        """Test that successful image generation works normally."""
        test_image = Image.new("RGB", (100, 100), color="red")
        record = ImageRecord(
            image=test_image,
            prompt="test prompt",
            model="test-model",
        )

        assert record.image is not None
        assert record.image == test_image
        assert record.error is None


class TestImageRecordRefusalEdgeCases:
    """Test edge cases in refusal detection."""

    def test_refusal_detection_with_partial_keyword_match(self):
        """Test that partial keyword matches work (e.g., 'refus' matches 'refused')."""
        assert ImageRecord._is_refusal_error("The request was refused")
        assert ImageRecord._is_refusal_error("Refusal of service")

    def test_cannot_generate_is_treated_as_refusal(self):
        """Test that 'cannot generate' is treated as refusal (usually policy-based)."""
        error_record = ImageRecord(
            image=None,
            error="Cannot generate image for this prompt",
            prompt="test prompt",
            model="test-model",
        )
        # Should use sad robot since "cannot generate" is policy-related
        assert error_record.image is not None

    def test_unable_to_generate_is_treated_as_refusal(self):
        """Test that 'unable to generate' is treated as refusal (usually policy-based)."""
        error_record = ImageRecord(
            image=None,
            error="Unable to generate: prompt violates policy",
            prompt="test prompt",
            model="test-model",
        )
        # Should use sad robot
        assert error_record.image is not None

    def test_whitespace_in_error_message(self):
        """Test refusal detection with extra whitespace."""
        assert ImageRecord._is_refusal_error("  refusal  ")
        assert ImageRecord._is_refusal_error("content policy violation")

    def test_special_characters_in_error_message(self):
        """Test refusal detection with special characters."""
        assert ImageRecord._is_refusal_error("Refusal: content policy!")
        assert ImageRecord._is_refusal_error("Error: 'blocked' from safety filter")

    def test_multiple_refusal_keywords(self):
        """Test error message with multiple refusal indicators."""
        error_message = (
            "Request was refused due to content policy violation. "
            "The prompt contains unsafe content that is not allowed."
        )
        assert ImageRecord._is_refusal_error(error_message)

    def test_dict_error_with_missing_message_field(self):
        """Test dict error without 'message' field."""
        error_dict = {"type": "policy_error", "code": 403}
        # Should not be considered refusal if no message field
        assert not ImageRecord._is_refusal_error(error_dict)

    def test_dict_error_with_other_fields_containing_refusal(self):
        """Test dict error with refusal keyword in non-message field."""
        error_dict = {
            "message": "API error",
            "reason": "Refusal",  # But this is not the 'message' field
        }
        # Only 'message' field is checked, so this should not be detected as refusal
        assert not ImageRecord._is_refusal_error(error_dict)
