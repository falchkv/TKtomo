# Feature isolation (`tktomo-feature-isolation`)

Track one feature through a projection stack by hand and export a
fixed-size crop window that follows it. The exported file is the intended
input for the track-model app when the full frame is too cluttered to
label comfortably: labels placed on the crop map back to raw detector
coordinates through the metadata it carries.

## Workflow

1. **Load stack.** Recognized HDF5 stacks (datasets `proj` + `theta_rad`,
   as written by the slogger preprocessing or by this app) load with their
   binning and crop provenance from the file attrs. Anything else goes
   through a dataset browser (HDF5) or an angles dialog (TIFF), followed by
   a provenance dialog where you type the binning and raw-pixel crop of the
   loaded grid. Those numbers are what make exported values valid at full
   resolution, so the dialog is never skipped.
2. **Place keyframes.** Left click (or Space) marks the feature in the
   current view, one keyframe per view. Arrows step views, PageUp/Down by
   ten. Delete removes the current view's keyframe. A handful of keyframes
   spread over the scan is enough.
3. **Interpolation.** Between keyframes, u follows a fitted sinusoid
   `a*cos + b*sin + c` (what a rigid point on a rotating object does; needs
   three keyframes, falls back to linear) and v follows linear or PCHIP
   interpolation. The readout shows the sinusoid's amplitude, center and
   rms, and the dashed curve in the image is the interpolated trajectory.
4. **Crop window.** Set the window height and width; the cyan rectangle
   previews it in every view. Windows are clamped at the stack edges.
5. **Export.** Writes an HDF5 with datasets `proj` (n, h, w), `theta_rad`,
   `crop_origin` (n, 2, the per-view window origin) and attrs
   `tktomo_feature_crop`, `binning`, `crop`, `window`, `source`,
   `keyframes`, `u_mode`, `v_mode`.

Keyframes can be saved to and reloaded from JSON (Save/Load keyframes),
including the window size and interpolation modes.

## Coordinate guarantee

For a pixel visible in both the original stack and the crop,

    raw(u_crop + origin_u[view], v_crop + origin_v[view])  ==  raw(u, v)

where `raw()` is the composition `crop_offset + index * binning +
(binning - 1) / 2`. The round trip is pinned by
`tests/test_tracking_io.py::test_feature_crop_round_trip`.
