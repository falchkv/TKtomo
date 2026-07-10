"""A curated, named colormap registry that returns pyqtgraph-ready LUTs.

Colormaps are drawn from matplotlib, colorcet, cmocean and cmasher. Source
libraries are imported lazily and only maps whose source is installed are
advertised by :func:`available_colormaps`, so the UI dropdown never lists a map
it cannot build. Grayscale is always available as a dependency-free fallback.

A curated entry is ``display_name -> (source, source_name)``.
"""

from __future__ import annotations

import numpy as np

_N = 256

# Curated selection. Perceptually-uniform, colour-vision-deficiency friendly.
_CURATED: dict[str, tuple[str, str]] = {
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


def _load_lut(source: str, source_name: str) -> np.ndarray:
    """Return an ``(_N, 3)`` uint8 RGB lookup table, or raise ImportError."""
    samples = np.linspace(0.0, 1.0, _N)
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
