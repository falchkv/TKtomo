"""A curated, named colormap registry that returns pyqtgraph-ready LUTs.

Colormaps are drawn from matplotlib, colorcet, cmocean and cmasher. Source
libraries are imported lazily and only maps whose source is installed are
advertised by :func:`available_colormaps`, so the UI dropdown never lists a map
it cannot build. Grayscale is always available as a dependency-free fallback.

A curated entry is ``display_name -> (source, source_name)``. The
``builtin`` source is computed here with numpy alone (grayscale and the
DECTRIS ALBULA scales), so those names are always available.
"""

from __future__ import annotations

import numpy as np

_N = 256

# Curated selection. Perceptually-uniform, colour-vision-deficiency friendly.
_CURATED: dict[str, tuple[str, str]] = {
    # dependency-free, computed below
    "albula": ("builtin", "albula_hdr"),
    "albula-hot": ("builtin", "albula_hot"),
    # matplotlib built-ins (uniform)
    "viridis": ("matplotlib", "viridis"),
    "plasma": ("matplotlib", "plasma"),
    "inferno": ("matplotlib", "inferno"),
    "magma": ("matplotlib", "magma"),
    "cividis": ("matplotlib", "cividis"),
    # colorcet (Kovesi)
    "cet-fire": ("colorcet", "fire"),
    "cet-rainbow": ("colorcet", "rainbow"),
    "cet-diverging-bwr": ("colorcet", "diverging_bwr_40_95_c42"),
    # cmocean
    "cmo-balance": ("cmocean", "balance"),
    "cmo-thermal": ("cmocean", "thermal"),
    # cmasher
    "cmr-chroma": ("cmasher", "chroma"),
    "cmr-rainforest": ("cmasher", "rainforest"),
}


def _grayscale_lut() -> np.ndarray:
    ramp = np.linspace(0, 255, _N, dtype=np.uint8)
    return np.stack([ramp, ramp, ramp], axis=1)


def _albula_hot_ramp(s: np.ndarray) -> np.ndarray:
    """Black to red to orange to yellow to white over ``s`` in [0, 1].

    Breakpoints measured from the gradient bar of the ALBULA histogram
    widget: red saturates 30 % in, green at 63 %, blue starts at 67 %.
    """
    r = np.clip(s / 0.30, 0.0, 1.0)
    g = np.clip((s - 0.30) / (0.63 - 0.30), 0.0, 1.0)
    b = np.clip((s - 0.67) / (1.0 - 0.67), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _albula_lut(hdr: bool) -> np.ndarray:
    """The DECTRIS ALBULA "High Dynamic Range" scale as an RGB table.

    Two ramps glued together: an inverted grey wedge (white at the bottom
    of the scale falling to black at the mid point) followed by the hot
    wedge. Weak signal reads as dark streaks on white, strong signal as a
    red to yellow core. ``hdr=False`` returns only the hot wedge.
    """
    t = np.linspace(0.0, 1.0, _N)
    if not hdr:
        rgb = _albula_hot_ramp(t)
    else:
        split = 0.5
        rgb = np.zeros((_N, 3))
        grey = t < split
        rgb[grey] = (1.0 - t[grey] / split)[:, None]
        rgb[~grey] = _albula_hot_ramp((t[~grey] - split) / (1.0 - split))
    return (rgb * 255).astype(np.uint8)


_BUILTIN = {
    "albula_hdr": lambda: _albula_lut(hdr=True),
    "albula_hot": lambda: _albula_lut(hdr=False),
}


def _load_lut(source: str, source_name: str) -> np.ndarray:
    """Return an ``(_N, 3)`` uint8 RGB lookup table, or raise ImportError."""
    samples = np.linspace(0.0, 1.0, _N)
    if source == "builtin":
        return _BUILTIN[source_name]()
    if source == "matplotlib":
        import matplotlib  # noqa: PLC0415

        cmap = matplotlib.colormaps[source_name]
        rgb = np.asarray(cmap(samples))[:, :3]
    elif source == "colorcet":
        import colorcet  # noqa: PLC0415

        # colorcet exposes lists of hex strings and matplotlib cmaps as cm[name].
        cmap = colorcet.cm[source_name]
        rgb = np.asarray(cmap(samples))[:, :3]
    elif source in ("cmocean", "cmasher"):
        import importlib  # noqa: PLC0415

        mod = importlib.import_module(f"{source}.cm" if source == "cmocean" else source)
        cmap = getattr(mod, source_name)
        rgb = np.asarray(cmap(samples))[:, :3]
    else:  # pragma: no cover - guarded by curated table
        raise ValueError(f"Unknown colormap source: {source}")
    return (rgb * 255).astype(np.uint8)


def available_colormaps() -> list[str]:
    """Names whose source library is importable (grayscale always included)."""
    names = ["grayscale"]
    for name, (source, _sn) in _CURATED.items():
        try:
            if source != "builtin":
                __import__(source)
            names.append(name)
        except ImportError:
            continue
    return names


def get_lut(name: str) -> np.ndarray:
    """Return an ``(256, 3)`` uint8 RGB lookup table for ``name``."""
    if name == "grayscale":
        return _grayscale_lut()
    if name not in _CURATED:
        raise KeyError(f"Unknown colormap {name!r}. Available: {available_colormaps()}")
    source, source_name = _CURATED[name]
    return _load_lut(source, source_name)


def get_colormap(name: str):
    """Return a ``pyqtgraph.ColorMap`` for ``name`` (requires pyqtgraph)."""
    import pyqtgraph as pg  # noqa: PLC0415

    lut = get_lut(name)
    positions = np.linspace(0.0, 1.0, len(lut))
    return pg.ColorMap(positions, lut)
