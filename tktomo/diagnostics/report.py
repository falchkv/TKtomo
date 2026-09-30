"""Rendering a :class:`~tktomo.diagnostics.artifacts.Verdict`: text, JSON, and a figure.

The text report is the primary output and is deliberately plain: it prints the ranked
findings with the numbers that produced them and the fix, then a log line for *every*
probe including the ones that did not run and why. That last part is the point --
a diagnosis that only shows what fired lets a stack of ``NOT_APPLICABLE`` probes read
as a clean bill of health.

``matplotlib`` is imported inside :func:`plot_verdict` only, so importing this module
(and therefore the whole ``tktomo.diagnostics`` package) needs numpy alone.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tktomo.diagnostics.artifacts import CATALOGUE, FailureMode, Verdict

__all__ = ["format_catalogue", "format_verdict", "plot_verdict", "save_verdict"]

_RULE = "=" * 88
_THIN = "-" * 88


def _wrap(text: str, indent: int, width: int) -> str:
    pad = " " * indent
    return "\n".join(textwrap.wrap(text, width=width, initial_indent=pad, subsequent_indent=pad))


def _bar(confidence: float, cells: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, confidence)) * cells))
    return "#" * filled + "." * (cells - filled)


def format_verdict(verdict: Verdict, *, width: int = 88, show_probes: bool = True) -> str:
    """Human-readable report: header, ranked findings, then every probe's status."""
    ctx = dict(verdict.context)
    lines: list[str] = [_RULE]
    shape = f"{int(ctx.get('n_theta', 0))} projections, " f"{int(ctx.get('n_v', 0))} x {int(ctx.get('n_u', 0))} px"
    lines.append(f"ARTIFACT DIAGNOSIS -- {shape}, {ctx.get('span_deg', float('nan')):.1f} deg span")
    notes = [
        f"angles read as {ctx.get('theta_units', '?')}",
        f"assumed centre {ctx.get('assumed_center_px', float('nan')):.2f} px",
        "sign flipped so mass is positive" if ctx.get("sign_inverted") else "sign as supplied",
    ]
    if ctx.get("volume_supplied"):
        notes.append("volume supplied")
    lines.append("  " + " | ".join(notes))
    ran = sum(1 for p in verdict.probes if p.ran)
    lines.append(
        f"  {ran} of {len(verdict.probes)} probes ran (coverage {verdict.coverage:.0%})"
        + (f" | triage stopped at: {verdict.stopped_at}" if verdict.stopped_at else "")
    )
    lines.append(_RULE)
    lines.append("")

    if not verdict.findings:
        lines.append("RANKED FINDINGS: none -- no probe fired.")
        if verdict.coverage < 1.0:
            lines.append(
                _wrap(
                    "Read that with the probe log below: a probe that could not run has "
                    "found nothing, which is not the same as finding nothing wrong.",
                    2,
                    width,
                )
            )
    else:
        lines.append("RANKED FINDINGS")
        for i, finding in enumerate(verdict.findings, 1):
            spec = finding.spec
            lines.append(
                f" {i}. [{finding.confidence:0.2f}] {_bar(finding.confidence)}  "
                f"mode {spec.number:>2d}  {spec.title}"
            )
            lines.append(f"        probe {finding.probe} | stage {spec.stage.value}")
            lines.append(_wrap(finding.detail, 8, width))
            if finding.evidence:
                evidence = ", ".join(
                    f"{k}={_fmt(v)}" for k, v in finding.evidence.items()
                )
                lines.append(_wrap(f"evidence: {evidence}", 8, width))
            lines.append(_wrap(f"slice signature: {spec.slice_signature}", 8, width))
            lines.append(_wrap(f"FIX: {spec.fix}", 8, width))
            lines.append("")

    if show_probes:
        lines.append(_THIN)
        lines.append("PROBE LOG")
        lines.append(f"  {'stage':<16}{'probe':<20}{'status':<16}note")
        for result in verdict.probes:
            note = result.reason or result.detail
            lines.append(
                f"  {result.stage.value:<16}{result.probe:<20}{result.status.value.upper():<16}"
                + textwrap.shorten(note, width=max(20, width - 54), placeholder=" ...")
            )
        lines.append(_THIN)
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return "nan"
    if val != 0 and (abs(val) < 1e-3 or abs(val) >= 1e5):
        return f"{val:.3e}"
    return f"{val:.4g}"


def format_catalogue(
    *, width: int = 88, modes: Iterable[FailureMode | str] | None = None
) -> str:
    """The 12-row artifact-to-cause table as text, for a doc or a terminal."""
    chosen = (
        sorted(CATALOGUE.values(), key=lambda s: s.number)
        if modes is None
        else [CATALOGUE[FailureMode(m)] for m in modes]
    )
    lines = [_RULE, "ARTIFACT -> CAUSE", _RULE]
    for spec in chosen:
        lines.append(f"{spec.number:>2d}. {spec.title}   [{spec.mode.value}, stage {spec.stage.value}]")
        lines.append(_wrap(f"slice:    {spec.slice_signature}", 4, width))
        lines.append(_wrap(f"sinogram: {spec.sinogram_signature}", 4, width))
        lines.append(_wrap(f"confirm:  {spec.confirm}", 4, width))
        lines.append(_wrap(f"fix:      {spec.fix}", 4, width))
        if spec.notes:
            lines.append(_wrap(f"note:     {spec.notes}", 4, width))
        lines.append("")
    return "\n".join(lines)


def save_verdict(verdict: Verdict, path: str | Path, *, include_curves: bool = False) -> Path:
    """Write the verdict as JSON. Returns the path written.

    ``include_curves`` adds the per-projection arrays (vacuum offset, vertical shift,
    centroid residual, ...). They are excluded by default: at a thousand projections and a
    dozen curves that is a data file, not a report.
    """
    path = Path(path)
    path.write_text(verdict.to_json(include_curves=include_curves), encoding="utf-8")
    return path


#: Which probe curve goes in which panel of :func:`plot_verdict`, in order.
_PANELS: tuple[tuple[str, str, str, str, str], ...] = (
    ("vacuum_phase", "vacuum_offset", "projection", "vacuum offset (rad)", "mode 11: vacuum phase"),
    ("vacuum_phase", "ramp_pv", "projection", "ramp p-v (rad)", "mode 11: residual ramp"),
    ("truncation", "u_edge_left", "projection", "edge / peak", "mode 12: profile edges"),
    ("vertical_drift", "vertical_shift_px", "projection", "shift (px)", "mode 3: vertical shift"),
    ("shift_jitter", "horizontal_residual_px", "projection", "residual (px)", "mode 2: centroid residual"),
    ("axis_tilt", "band_c", "band height z (px)", "axis position (px)", "modes 4/5/6: arc test"),
    ("center_sweep", "entropy", "assumed centre (px)", "slice entropy", "mode 1: centre sweep"),
    ("scale_drift", "scale_series", "projection", "relative scale", "mode 8: scale drift"),
    ("deformation", "residual_frac", "angle index", "residual / rms", "mode 9: reprojection residual"),
)


def plot_verdict(verdict: Verdict, *, path: str | Path | None = None, max_panels: int = 6):
    """Optional matplotlib figure: confidence bars plus the curves behind them.

    Returns the ``Figure``. Raises ``ImportError`` with an actionable message when
    matplotlib is missing -- the text report needs nothing but numpy, and this is the
    only part of the package that does not.
    """
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "plot_verdict needs matplotlib (pip install matplotlib). The text report "
            "(format_verdict) and the JSON (save_verdict) need only numpy."
        ) from exc

    panels = []
    for probe_name, key, xlabel, ylabel, title in _PANELS:
        try:
            result = verdict.probe(probe_name)
        except KeyError:
            continue
        if key in result.curves and np.asarray(result.curves[key]).size:
            panels.append((result, key, xlabel, ylabel, title))
        if len(panels) >= max_panels:
            break

    rows = 1 + int(np.ceil(len(panels) / 2))
    fig = plt.figure(figsize=(11, 2.1 + 2.4 * (rows - 1)))
    grid = fig.add_gridspec(rows, 2, height_ratios=[1.1] + [1.0] * (rows - 1))

    top = fig.add_subplot(grid[0, :])
    if verdict.findings:
        # Numeric positions, not category labels: two probes can report the SAME mode
        # (mode 1 has two independent probes) and a categorical bar chart silently
        # stacks them on one bar, hiding the corroboration that is the whole point.
        labels = [f"{f.spec.number}. {f.mode.value}  ({f.probe})" for f in verdict.findings][::-1]
        values = [f.confidence for f in verdict.findings][::-1]
        y = np.arange(len(values))
        top.barh(y, values, color="#b03030")
        top.set_yticks(y)
        top.set_yticklabels(labels, fontsize=8)
        top.set_xlim(0, 1)
        top.set_xlabel("confidence (monotone in evidence, not a probability)")
    else:
        top.text(0.5, 0.5, "no probe fired", ha="center", va="center")
        top.set_xticks([])
        top.set_yticks([])
    ran = sum(1 for p in verdict.probes if p.ran)
    stopped = f", triage stopped at {verdict.stopped_at}" if verdict.stopped_at else ""
    top.set_title(
        f"artifact diagnosis -- {ran}/{len(verdict.probes)} probes ran{stopped}", loc="left"
    )

    for i, (result, key, xlabel, ylabel, title) in enumerate(panels):
        ax = fig.add_subplot(grid[1 + i // 2, i % 2])
        y = np.asarray(result.curves[key], dtype=float)
        if result.probe == "axis_tilt":
            ax.plot(np.asarray(result.curves["band_z"], dtype=float), y, "o-")
        elif result.probe == "center_sweep":
            ax.plot(np.asarray(result.curves["sweep_center_px"], dtype=float), y, "-")
        else:
            ax.plot(y, lw=0.8)
            if result.probe == "vertical_drift" and "vertical_trend_px" in result.curves:
                ax.plot(np.asarray(result.curves["vertical_trend_px"], dtype=float), "r-", lw=1.5)
            if result.probe == "truncation" and "u_edge_right" in result.curves:
                ax.plot(np.asarray(result.curves["u_edge_right"], dtype=float), lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, loc="left")

    fig.tight_layout()
    if path is not None:
        fig.savefig(Path(path), dpi=130, bbox_inches="tight")
    return fig
