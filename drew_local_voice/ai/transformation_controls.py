"""Bounded, preview-only controls for adaptive AI transformations.

The UI passes these values to the existing PreviewGate path.  This module is
pure so the contract can be validated without loading Tk, audio, or a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

AMPLIFICATION_CHOICES: tuple[float, ...] = (1.0, 2.0, 3.0)
STYLE_CHOICES: tuple[str, ...] = (
    "Preserve voice",
    "Formal",
    "Casual",
    "Technical",
    "Creative",
)
EXECUTION_CHOICES: tuple[str, ...] = ("current", "modified", "parallel", "batch")


@dataclass(frozen=True)
class TransformationControls:
    """User-selected transformation modifiers with safe bounded defaults."""

    amplification: float = 1.0
    style: str = "Preserve voice"
    length_scale: float = 1.0
    execution: str = "current"

    def normalized(self) -> "TransformationControls":
        return TransformationControls(
            amplification=normalize_amplification(self.amplification),
            style=normalize_style(self.style),
            length_scale=normalize_length_scale(self.length_scale),
            execution=normalize_execution(self.execution),
        )

    def as_dict(self) -> dict[str, Any]:
        value = self.normalized()
        return {
            "amplification": value.amplification,
            "style": value.style,
            "length_scale": value.length_scale,
            "execution": value.execution,
        }


def normalize_amplification(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 1.0
    return round(min(4.0, max(1.0, numeric)), 2)


def normalize_length_scale(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 1.0
    return round(min(2.0, max(0.5, numeric)), 2)


def normalize_style(value: Any) -> str:
    candidate = str(value or "").strip()
    return candidate if candidate in STYLE_CHOICES else STYLE_CHOICES[0]


def normalize_execution(value: Any) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in EXECUTION_CHOICES else "current"


def build_modifier_contract(
    controls: TransformationControls | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a bounded prompt contract; never includes source text.

    ``parallel`` and ``batch`` are deliberately bounded variants.  They remain
    inside the existing review card and never stage/replace/copy automatically.
    """
    if controls is None:
        value = TransformationControls()
    elif isinstance(controls, TransformationControls):
        value = controls
    else:
        value = TransformationControls(
            amplification=controls.get("amplification", 1.0),
            style=controls.get("style", STYLE_CHOICES[0]),
            length_scale=controls.get("length_scale", 1.0),
            execution=controls.get("execution", "current"),
        )
    value = value.normalized()
    variant_count = {"current": 1, "modified": 1, "parallel": 2, "batch": 3}[value.execution]
    return {
        **value.as_dict(),
        "variant_count": variant_count,
        "instruction_suffix": _instruction_suffix(value, variant_count),
        "system_suffix": _system_suffix(value, variant_count),
        "output_multiplier": round(
            min(6.0, max(1.0, value.amplification * value.length_scale * variant_count)),
            2,
        ),
    }


def _instruction_suffix(value: TransformationControls, variant_count: int) -> str:
    parts: list[str] = []
    if value.amplification > 1.0:
        parts.append(
            f"Amplification target: {value.amplification:g}x. Add proportional specificity and useful detail without inventing facts."
        )
    if value.style != STYLE_CHOICES[0]:
        parts.append(f"Style target: {value.style}. Preserve the source meaning and grounded facts.")
    if value.length_scale != 1.0:
        direction = "shorter" if value.length_scale < 1.0 else "longer"
        parts.append(f"Length target: make the result approximately {value.length_scale:g}x as {direction} as the source while keeping it coherent.")
    if value.execution == "parallel":
        parts.append("Return exactly two clearly labeled alternatives for side-by-side review.")
    elif value.execution == "batch":
        parts.append("Return exactly three clearly labeled variants for batch review.")
    return "\n".join(parts)


def _system_suffix(value: TransformationControls, variant_count: int) -> str:
    if value.execution in {"parallel", "batch"}:
        return (
            f"This is a bounded {value.execution} preview with {variant_count} variants. "
            "Do not execute actions, send messages, or apply any variant."
        )
    return "This is a review-only modified preview. Do not execute actions or apply the result."


__all__ = [
    "AMPLIFICATION_CHOICES",
    "EXECUTION_CHOICES",
    "STYLE_CHOICES",
    "TransformationControls",
    "build_modifier_contract",
    "normalize_amplification",
    "normalize_execution",
    "normalize_length_scale",
    "normalize_style",
]
