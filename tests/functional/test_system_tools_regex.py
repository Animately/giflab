"""Unit tests for the ``_extract_version`` regex helper in ``giflab.system_tools``.

``_extract_version(output, pattern)`` is a pure helper used by the tool-discovery
code to pull a version string out of a binary's ``--version`` output. It runs
``re.search(pattern, output)`` and returns the first capture group, or ``None``
when there is no match. These tests cover that contract directly, independent of
any external binary.
"""

import pytest
from giflab.system_tools import _extract_version


class TestExtractVersionMatching:
    """Cases where the pattern matches and a capture group is returned."""

    def test_returns_first_capture_group(self):
        assert _extract_version("version=1.2.3", r"version=(\d+\.\d+\.\d+)") == "1.2.3"

    def test_returns_group_one_when_pattern_has_multiple_groups(self):
        # group(1) is returned even though group(2) also matches.
        result = _extract_version("release 4-beta", r"release (\d+)-(\w+)")
        assert result == "4"

    def test_match_anywhere_in_output_not_just_start(self):
        # re.search (not re.match) means a match mid-string still counts.
        assert _extract_version("prefix noise gifski 1.94", r"gifski (\S+)") == "1.94"

    def test_multiline_output_matches_on_later_line(self):
        output = "first line\nffmpeg version 6.1.1\nthird line"
        assert _extract_version(output, r"ffmpeg version (\S+)") == "6.1.1"

    def test_first_of_multiple_matches_is_returned(self):
        # re.search stops at the first match; the second "9.9" is ignored.
        assert _extract_version("v1.0 then v9.9", r"v(\d+\.\d+)") == "1.0"


class TestExtractVersionRealisticToolLines:
    """Realistic tool-version output lines with a semver-style pattern."""

    SEMVER = r"(\d+\.\d+(?:\.\d+)?)"

    def test_ffmpeg_version_line(self):
        assert _extract_version("ffmpeg version 6.1.1", self.SEMVER) == "6.1.1"

    def test_gifsicle_two_component_version(self):
        # Trailing optional ".\d+" is absent here, so a 2-component match returns.
        assert _extract_version("gifsicle 1.94", self.SEMVER) == "1.94"

    def test_imagemagick_named_pattern(self):
        assert (
            _extract_version("Version: ImageMagick 7.1.1-15", r"ImageMagick (\S+)")
            == "7.1.1-15"
        )

    def test_animately_version_label(self):
        assert _extract_version("Version: 1.1.20.0", r"Version: (\S+)") == "1.1.20.0"


class TestExtractVersionNoMatch:
    """Cases where the helper must return ``None``."""

    def test_no_match_returns_none(self):
        assert _extract_version("no version here", r"version (\d+)") is None

    def test_empty_output_returns_none(self):
        assert _extract_version("", r"(\d+\.\d+)") is None

    def test_pattern_present_but_capture_group_unmatched_portion(self):
        # The literal text exists but the digit group does not follow it.
        assert _extract_version("ffmpeg version", r"ffmpeg version (\d+)") is None


class TestExtractVersionSpecialCharacters:
    """Patterns containing regex metacharacters that must be matched literally."""

    def test_escaped_parentheses_in_pattern(self):
        # Literal parens around the build, capture group pulls the inner version.
        result = _extract_version("tool (build 2.5)", r"\(build (\d+\.\d+)\)")
        assert result == "2.5"

    def test_escaped_dot_and_plus_metacharacters(self):
        # The '+' and '.' in the output are matched via escaped metacharacters.
        result = _extract_version("c++ 14.2.0", r"c\+\+ (\d+\.\d+\.\d+)")
        assert result == "14.2.0"

    def test_dollar_anchor_matches_version_at_end(self):
        assert _extract_version("up to date: 3.0.1", r"(\d+\.\d+\.\d+)$") == "3.0.1"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
