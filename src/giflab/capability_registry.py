from __future__ import annotations

"""Capability registry – maps compression variables to available tool wrappers.

Stage-3 requirement: enumerate every tool that can fulfil *one* variable slot so
that single-variable strategies can be generated automatically.
"""

import inspect
from abc import ABC
from collections.abc import Iterator

from .tool_interfaces import (
    ColorReductionTool,
    ExternalTool,
    FrameReductionTool,
    LossyCompressionTool,
)

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _iter_all_wrapper_subclasses() -> Iterator[type[ExternalTool]]:
    """Yield every subclass defined in *giflab.tool_wrappers* (import side-effect)."""
    from . import tool_wrappers  # noqa: F401 – ensures module is imported

    for _, obj in inspect.getmembers(tool_wrappers, inspect.isclass):
        if issubclass(obj, ExternalTool) and obj is not ExternalTool:
            # Comprehensive base class filtering
            if _is_base_class(obj):
                # Optional debug logging (can be enabled for troubleshooting)
                # import logging
                # logging.getLogger(__name__).debug(f"Filtered base class: {obj.__name__}")
                continue

            yield obj


def _is_base_class(cls: type) -> bool:
    """Comprehensive check to determine if a class is a base class rather than a concrete tool.

    Args:
        cls: Class to check

    Returns:
        True if the class should be excluded as a base class
    """
    # 1. Known base classes by type
    EXCLUDED_BASE_CLASSES = {
        ExternalTool,
        ColorReductionTool,
        FrameReductionTool,
        LossyCompressionTool,
    }

    if cls in EXCLUDED_BASE_CLASSES:
        return True

    # 2. Abstract base classes (ABC)
    if getattr(cls, "__abstractmethods__", None):
        return True

    # 3. Classes that directly inherit from ABC (not through tool interfaces)
    # Only filter classes that directly inherit from ABC, not through tool interfaces
    if ABC in cls.__bases__:
        return True

    # 4. Classes with underscore prefix (naming convention for base classes)
    if cls.__name__.startswith("_"):
        return True

    # 5. Classes with "base" in the name (case-insensitive)
    if "base" in cls.__name__.lower():
        return True

    # 6. Classes with generic/template naming patterns
    generic_patterns = ["template", "mixin", "abstract", "interface", "proto"]
    if any(pattern in cls.__name__.lower() for pattern in generic_patterns):
        return True

    # 7. Check NAME attribute for base class indicators
    class_name = getattr(cls, "NAME", "")
    if class_name:
        base_name_patterns = {
            "external-tool",  # Known base class name
            "base-tool",
            "abstract-tool",
            "template-tool",
        }

        if class_name in base_name_patterns:
            return True

        # Check for generic patterns in NAME
        if any(
            pattern in class_name.lower()
            for pattern in ["base", "abstract", "template"]
        ):
            return True

    # 8. Check for missing required concrete implementations
    # A concrete tool should have a valid NAME and VARIABLE
    if not hasattr(cls, "NAME") or not getattr(cls, "NAME", "").strip():
        return True

    if not hasattr(cls, "VARIABLE") or not getattr(cls, "VARIABLE", "").strip():
        return True

    # 9. Check docstring for base class indicators
    docstring = cls.__doc__ or ""
    base_doc_patterns = [
        "base class",
        "abstract class",
        "template class",
        "mixin",
        "interface",
    ]
    if any(pattern in docstring.lower() for pattern in base_doc_patterns):
        return True

    # 10. Check for placeholder/stub implementations
    # Look for methods that just raise NotImplementedError
    for method_name in ["apply", "available"]:
        if hasattr(cls, method_name):
            method = getattr(cls, method_name)
            if hasattr(method, "__code__"):
                # Check if method just raises NotImplementedError
                try:
                    source_lines = inspect.getsourcelines(method)[0]
                    source = "".join(source_lines).strip()
                    if (
                        "notimplementederror" in source.lower()
                        and len(source_lines) <= 3
                    ):
                        return True
                except (OSError, TypeError):
                    pass  # Can't get source, skip this check

    # If none of the base class indicators are found, it's likely a concrete class
    return False


_REGISTRY: dict[str, list[type[ExternalTool]]] = {
    "color_reduction": [],
    "frame_reduction": [],
    "lossy_compression": [],
}

for cls in _iter_all_wrapper_subclasses():
    variable = getattr(cls, "VARIABLE", None)
    if variable in _REGISTRY and cls.available():
        _REGISTRY[variable].append(cls)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tools_for(variable: str) -> list[type[ExternalTool]]:
    """Return *available* tool wrapper classes for *variable*."""
    if variable not in _REGISTRY:
        raise ValueError(f"Unknown variable: {variable}")
    return list(_REGISTRY[variable])  # return copy


def all_single_variable_strategies() -> list[str]:
    """Return names like ``gifsicleColor`` representing each (variable, tool)."""
    names: list[str] = []
    for var, classes in _REGISTRY.items():
        for cls in classes:
            # Simple name scheme: <tool-name>|<variable-suffix> (camel-case)
            suffix = {
                "color_reduction": "Color",
                "frame_reduction": "Frame",
                "lossy_compression": "Lossy",
            }[var]
            names.append(f"{cls.NAME}_{suffix}")
    return names


def get_tool_class_by_name(name: str) -> type[ExternalTool] | None:
    """Look up a tool wrapper class by its NAME attribute.

    Args:
        name: The tool NAME string (e.g., ``"gifsicle-lossy"``, ``"animately-standard"``)

    Returns:
        The tool wrapper class, or ``None`` if not found.
    """
    for classes in _REGISTRY.values():
        for cls in classes:
            if cls.NAME == name:
                return cls
    return None
