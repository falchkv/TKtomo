"""Registry of apex detectors for auto-completing sphere tangent features.

The 1D matcher in `edgetrack` follows an edge by correlation: it knows
nothing about what an apex IS, only that the pixels around it look like
the pixels around the seed. A detector trained on a particular sample
knows more, and where one exists it should be usable instead.

So a detector is a plugin, selected by name through a registry, the same
way aligners, reconstruction backends and colormaps already are here. The
window lists `available_apex_detectors()`, so a registered detector
appears in the dropdown with no UI change, and the auto-track job carries
only the NAME across the wire, which the server resolves in its own
registry. Nothing about a detector has to be serialisable.

A detector for a particular sample does not belong in this package. Put
it in the analysis that owns the sample, and name its module in the
`TKTOMO_APEX_PLUGINS` environment variable (comma separated, on the
PYTHONPATH). That is read once, on the first registry query, on the
laptop and on the compute node alike.

Two things a detector must be honest about. `native_bin` is the mean-pool
factor OF THE FILE'S GRID that its scale parameters were tuned for, and
`run_autotrack` tracks at that factor rather than the one the feature
size implies, because a detector with absolute pixel scales silently
misbehaves on the wrong grid. It cannot protect you from a file that is
already binned differently from the one it was trained on: the app has no
way to know that, and a detector should say in its description which
stack it was trained on.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np

#: column layout of what `ApexDetector.detect` returns
DETECTION_COLUMNS = ("u", "v", "score", "sign")

#: the sign each tangent kind rides on: the apex at the SMALLER coordinate
#: along the constrained axis is -1, the larger is +1 (`model.SIGN_U` and
#: `SIGN_V` say the same thing about the model's own kinds)
SIGN_LOW, SIGN_HIGH = -1.0, +1.0


@runtime_checkable
class ApexDetector(Protocol):
    """Finds silhouette tangent points in one frame.

    `name` is what the dropdown and the job carry. `native_bin` is the
    grid the detector wants, as a mean-pool factor of the file's grid.
    """

    name: str
    native_bin: int

    def detect(self, frame: np.ndarray, axis: str) -> np.ndarray:
        """(N, 4) of (u, v, score, sign) for one frame.

        `axis` is the coordinate the wanted tangents constrain: "u" for
        the left and right edges of a body, "v" for its top and bottom.
        `score` is a confidence in 0 to 1, which the "min p" box
        thresholds. `sign` is -1 for the apex at the smaller coordinate
        along `axis` and +1 for the larger.
        """
        ...


_REGISTRY: dict[str, ApexDetector] = {}
_PLUGINS_LOADED = False
_PLUGIN_PROBLEMS: list[str] = []


def register_apex_detector(detector: ApexDetector) -> None:
    """Add a detector under its own `name`, replacing one of that name."""
    if not isinstance(detector, ApexDetector):
        raise TypeError(
            "an apex detector needs a name, a native_bin and a detect method")
    _REGISTRY[str(detector.name)] = detector


def _load_plugins() -> list[str]:
    """Import the modules named in TKTOMO_APEX_PLUGINS, once.

    A module registers its detectors on import. Failures are reported,
    not raised: a missing plugin must not stop the window from opening,
    it must only be absent from the dropdown.
    """
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return _PLUGIN_PROBLEMS
    _PLUGINS_LOADED = True
    _PLUGIN_PROBLEMS.clear()
    import importlib  # noqa: PLC0415
    names = [n.strip() for n in os.environ.get("TKTOMO_APEX_PLUGINS", "")
             .split(",") if n.strip()]
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:                          # noqa: BLE001
            _PLUGIN_PROBLEMS.append(f"{name}: {exc}")
    return _PLUGIN_PROBLEMS


def available_apex_detectors() -> list[str]:
    """Registered detector names, plugins loaded on the first call."""
    _load_plugins()
    return sorted(_REGISTRY)


def plugin_problems() -> list[str]:
    """Why a named plugin is not in the list. Empty when all loaded."""
    return _load_plugins()


def get_apex_detector(name: str) -> ApexDetector:
    _load_plugins()
    try:
        return _REGISTRY[str(name)]
    except KeyError:
        raise KeyError(
            f"no apex detector named {name!r}; available: "
            f"{', '.join(sorted(_REGISTRY)) or 'none'}. A detector for your "
            f"sample is registered by naming its module in "
            f"TKTOMO_APEX_PLUGINS.") from None


def detections_array(rows) -> np.ndarray:
    """Coerce a detector's output to the (N, 4) contract."""
    out = np.asarray(rows, float)
    if out.size == 0:
        return np.zeros((0, 4))
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(
            f"an apex detector returns (N, 4) of {DETECTION_COLUMNS}, "
            f"got {out.shape}")
    return out
