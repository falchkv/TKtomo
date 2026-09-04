# Track model (`tktomo-track-model`)

Fit a tomography acquisition model from hand-labeled feature tracks, watch
the residuals respond to every parameter, and export the result as a model
file, a slogger-compatible shift table, ASTRA vector geometry, or an
aligned projection stack. Built for datasets where automatic alignment
fails and the missing information is long-baseline correspondence only a
human can supply.

## The model

For feature i at view j (angles theta in radians, coordinates in RAW
detector pixels):

    s  = a_i cos(theta_j) + b_i sin(theta_j)     across the beam
    t  = -a_i sin(theta_j) + b_i cos(theta_j)    along the beam
    u  = s + c(theta_j) + dx_j
    v  = y_i + alpha(theta_j) s + beta(theta_j) t + dy_j

`c` is the rotation-axis position, `alpha` the in-plane tilt, `beta` the
out-of-plane tilt, each a polynomial in angle of user-chosen degree
(constant by default). `dx`/`dy` are per-view shifts. The fit is two
linear IRLS solves (Huber weights), no nonlinear optimizer: every view is
tied to every other through shared features, so there is nothing to chain
and nothing to diverge.

Sometimes the whole object tilts at a view, which no smooth tilt and no
shift can express. Three optional per-view rotations describe that: the
beam frame (across-beam horizontal, beam, rotation axis) with the
detector, rotated relative to the object by the small vector
(rot horiz, rot beam, rot axis) in its own axes. The feature's
coordinates (s, t, y) become p' = R(-w) p, then the equations above use
s', t', y'. To first order

    s' = s + rot_axis t - rot_beam y
    t' = t - rot_axis s + rot_horiz y
    y' = y - rot_horiz t + rot_beam s

Rot axis is an increment of the projection angle (theta + rot axis, added
to the nominal angle, not replacing it), rot beam an in-plane rotation
of the image about the axis column, rot horiz an out-of-plane tilt. With
a rotation free the fit becomes a joint Gauss-Newton on every free
parameter (three passes), each angle under a Gaussian prior whose rms
you set, weighed against the label noise. Without them the fit is bit
for bit the two-stage one.

## Labeling

Digits 0 to 9 pick the active feature (the table picks any id), left click
or Space places it in the current view and advances N views (configurable),
Delete removes the nearest label in the view, arrows step views. With
"Follow the prediction after a click" ticked, each placed label also pans
the view so the model's predicted position of the active feature in the
next view sits at the centre, at the current zoom. It acts only once the
feature has more than four labels and a fit exists. Labels are
stored in raw coordinates through the stack's provenance chain, so they
survive reloading the same data at another binning or crop, including
feature-crop stacks from `tktomo-feature-isolation`.

The **bin** box next to the view slider mean-pools the projections by 1,
2, 4 or 8 at any time, without reloading. It applies where the pixels are
(on the server for a remote stack, so a frame shrinks by the factor
squared before it crosses the link) and to the view, the recon slice and
the aligned export, which all see the binned grid. Auto-complete does
not: it tracks on a grid of its own, chosen from the feature's size (see
"Tracking grid" below). Labels stay in raw pixels, so switching back and
forth loses nothing. Marker sizes and the recon row follow the grid. The
factor is saved with the session and restored on load.

The active feature's predicted cross is drawn larger and in its feature
color (others stay white). A toggle shows the active feature's labels
from OTHER frames as faint half-size circles, useful for judging where
the next click belongs. The "Worst outlier" button jumps to the label
with the largest residual and makes its feature active. "Next
unlabelled" steps forward (wrapping) to the next projection with no
label at all, and once every projection has one, to the next with the
fewest labels.

Each feature has an editable marker SIZE (image pixels) in the table.
Match it to the physical feature: circles are drawn at that diameter in
data coordinates, and the fit weights each feature's labels by 1/size,
since a click on a large diffuse feature localizes it less than one on a
small sharp feature. Uniform sizes mean unweighted.

Clicking a point in the reconstructed slice marks it as a magenta
diamond in the projection view, projected through the model so it
follows the object point across frames (Escape clears it). The
slice-to-object mapping is pinned against tomopy's grid convention by a
dedicated test.

## Auto-completing tracks

Label a feature manually in a handful of views (at least 2), then
"Auto-complete feature" (or "Auto-complete all" for every feature with
2+ manual labels) fills the views around each manual label by template
matching, on a background worker with progress and cancel. The design
follows what tilt-series tools (IMOD Beadtrack) and trackers (TrackMate,
KLT) converge on, tuned with numbers measured on this very dataset in
the earlier slogger pipeline:

- Your manual labels are the ANCHORS. The template is always cut at the
  nearest manual label and never updated: chained templates drift
  smoothly and plausibly (the dangerous kind of wrong), anchored ones
  decorrelate honestly (correlation 0.65 one view away, 0.23 at ten), so
  a track STOPS when the template stops resembling the image. More
  manual labels = longer reach; every ~10 views is a good rhythm.
- The search window sits on the sinusoid your manual labels imply. The
  matcher is Hann-windowed phase correlation with iterative re-cut
  (a single pass systematically underestimates motion by 13 percent)
  and reports a plain correlation coefficient as quality.
- Seeds on edge-like structure are refused up front (structure-tensor
  coherence > 0.4): a patch on a filament can slide along it with a
  confident correlation, and its position along the edge would be
  fiction. Measured on a particle riding on its sample's edge
  (lens1_v11_upper): with the gate off, those seeds produced labels 11 to
  16 raw px wrong that the confidence still passed. A refused seed marks
  a stretch of views that stays manual, and the report says which.
- The forward-backward check (off by default) tracks every accepted
  match back to its seed and drops it if the round trip lands more than
  1 track px off the seed or the backward correlation falls below 0.10,
  a correlation threshold separate from "min p". Measured on the lens1
  particle it halved the coverage (50 against 79 percent of the views)
  for 5 points of precision, whatever the correlation threshold: a
  template cut far from its seed does not land back on it. The residual
  rejection after the fit catches the same lock-ons. Turn it on when a
  lookalike sits next to the feature.

**Tracking grid.** The tracker works in pixels, so what a 40 px template
sees depends on the grid. On a display grid binned by 4 it would cover
160 raw px around a 12 px particle, which is all sample edges, and the
coherence gate would refuse every seed (that is exactly what happened
before this grid existed: 0 to 20 percent coverage against 79 percent on
the file's grid). Auto-complete therefore picks its own grid from the
feature's marker size, independent of the bin box: the coarsest
mean-pool of the file's grid on which the template still covers four
times the marker (a 12 px feature tracks at bin 1, a 25 px one at bin 2).
The label under the parameters says what the next run will do for the
active feature, and the "track bin" box overrides the choice. Set the
marker size to the feature's real size in the table first, since the
default 10 px is a guess. On the lens1 particle bin 2 was three times
faster than bin 1 at 2.2 against 1.4 raw px median error, same
precision, slightly less coverage. The first run on a grid high-passes
the whole stack there (about three minutes for 907 views of 557x1816 at
bin 1 on 8 cores, a quarter of that at bin 2), later runs reuse it. If
the high-passed copy does not fit in memory the host steps the bin up
and says so in the report.

The search radius is entered in raw px so it means the same at every
binning, and its maximum is what the template can see: 18 track px,
which the box converts.

Auto labels are drawn as HOLLOW circles (same color and size), carry
their match quality, and enter the fit at full weight: the Huber loop
and the Worst-outlier button are the review path. A manual click always
overwrites an auto label; the auto-tracker never touches a manual one;
re-running replaces only auto labels; "Clear auto" undoes the machine's
work in bulk. Not supported on per-view-cropped (feature-isolation)
stacks, whose windows already follow the feature.

A structural benefit: auto-completion populates many features across the
SAME views, so multi-label views become the norm and the
staggered-labeling trap (warning W6, free shifts absorbing single-label
views) largely disappears.

Defaults, all measured: min p 0.20, search radius 8 raw px (+0.25
track px per view from the seed, capped at 18 track px), patch 40 px on
the tracking grid, high-pass sigma 12 track px, forward-backward off
(when on: correlation 0.10 and round trip 1 px), coherence gate 0.4,
three consecutive failures end a direction. With those, seeds every ~50
views on the lens1 particle gave 79 percent coverage at 95 percent within
4 raw px, and seeds every ~200 views 74 percent at 97 percent.

**The report.** After a run the box under the status line keeps one
block per feature until the next run (the fit never overwrites it): the
grid tracked on, which seeds were used and which refused and why, how
many views each gate dropped ("no match", "below min p", "fwd-back
miss", "behind a stopped march"), and the largest unlabelled gap with
the remedy, usually a manual label in the middle of it.

**How a match is scored.** Completion uses a learned matcher (the earlier
single-anchor phase-correlation completer has been removed). For each
view the 5 manual labels nearest in angle each match independently, the
position is their quality-weighted median (same median error as one
anchor, half the p90), and a gradient-boosted classifier rates the answer
with the probability that it lies within 4 raw px, from 43 features
describing how the answer was reached (agreement of the five, the fused
correlation map, patch structure, residual against the sinusoid). On
held-out features that probability separates good from bad at AUC 0.92,
where a plain correlation sits at 0.6; the "min p" box thresholds it and
0.20 is the measured operating point. Measured end to end on the
graphite-ball stack: anchors every 10 views plus completion reproduce the
fully hand-labeled alignment to 1.4 raw px median in dx; 20-view anchors
are the floor, 30 does not work. Trained on the one available dataset, so
treat the probabilities as ranks until a second sample confirms them.
Auto-completion needs scikit-learn and joblib (in the `ui` extra); without
them the buttons report the matcher as unavailable. Roughly 20 s for the
whole scan on 8 cores.

**Residual rejection.** With "reject auto >" on (default, 3 x Huber), every
fit is followed by one pass that removes AUTO labels whose residual exceeds
that limit and refits. Manual labels are never removed. The Huber loop only
down-weights a wrong match, and in a view carried by two labels a
down-weighted 15 px lock-on still moves the free shift by several pixels;
removing it does not. This pass alone recovered most of the learned
matcher's end-to-end gain for the plain phase-correlation completer.

## Fixed and free parameters

Every polynomial coefficient has a "fix" checkbox and an editable value;
The out-of-plane tilt beta starts fixed at zero (few features constrain
it and it trades off against y and dy), untick "fix" to fit it.
`dx` and `dy` are fixed or freed as whole groups and start FIXED (at
zero) so the first fits explain the labels with axis geometry alone;
free them once several features share views. A feature row can be
pinned, which fixes its (a, b, y). Editing any value re-evaluates the
residuals WITHOUT fitting, so the response to "what if the center were
here" is immediate; the next fit overwrites free values and respects fixed
ones.

The three per-view rotations ("Rot horiz", "Rot beam", "Rot axis") work
the same way: a free box, a Zero button, and a prior rms in degrees
each. Unchecked means frozen at the current values, which are zero until
a fit or a session load sets them, so read the state line under the
rows. The prior keeps an angle small where few labels constrain it: a
view needs three or more labels at different positions for the angle to
be measured at all (warning W7 counts the thin views), and the "label
noise" box in the Robust row sets how strongly the priors pull (the fit
minimises the squared residuals plus noise squared times the squared
angles over their rms). One degree leaves a real degree-scale tilt alone
and damps the noise-driven ones. Above about ten degrees the prior stops
doing its job and the angles drift into directions the labels cannot
tell from feature heights and tilts, so the box stops there. Rot beam
shares its lever with dy and y, so free dx and dy with it. A constant
rot axis over all views is the same as rotating every feature's (a, b),
a constant rot beam the same as alpha, a constant rot horiz the same as
beta: the prior picks the representative with the angles at zero mean,
so the polynomial coefficients keep their meaning.

Two consequences of the geometry to keep in mind (the app warns about
both):

- With `dx` free, the constant/cos/sin part of the shifts is
  indistinguishable from the object sitting elsewhere. The app regauges
  it into (c, a, b) so the reported center is canonical, but on a short
  angular arc the center is still barely separable from the feature
  amplitudes (warning W1). Pinning one feature at known coordinates, or
  fixing `dx`, breaks the gauge and makes the center a measurement.
- A coefficient that is FIXED while its shift group is FREE cannot affect
  the fit; the shifts absorb it (warning W3). Fix the shifts too if you
  want the fixed value to mean something.

`Run diagnostics` fits disjoint halves of the features separately and
reports the disagreement (center, tilts, shift curves, the three rotation
curves in degrees) plus held-out residuals. The half-split is the number
to trust; the parametric sigma is shown but loses every argument with it.

## Plots and views without labels

The bottom panel holds one plot pane with a dropdown selecting what it
shows (default: labels per view). Available: dx/dy shifts, the three
per-view rotations in degrees, labels per view, residual u/v colored by
feature, per-view MAD spread, axis center c(theta), tilts alpha/beta,
residual histogram. Clicking in
any angle-axis plot jumps to the nearest frame, so a gap or a bad point
is one click from being looked at and labeled.

`dx`/`dy` and the rotations are measured only where labels exist;
unlabeled views are filled by interpolation over angle. In the shift
and rotation plots, dots are labeled
views (orange when only ONE label carries the view, so its shift is that
label verbatim), the dashed curve is the interpolation, and red base
ticks mark frames with NO labels at all. The "labels per view" plot
shows the coverage directly, with the same red ticks and an orange
guide line at two labels.

## Files

Loading a stack, saving and loading a session, and every export live in
the File menu (Ctrl+O stack, Ctrl+Shift+O session, Ctrl+S save session).

## Remote stack (`tktomo-track-server`)

Labelling needs many clicks but only one projection at a time, so for a
large dataset the stack can stay on the machine that holds it (a Maxwell
node) while the window runs natively on your laptop. A small server on the
node serves one frame per view change; the live recon slice, auto-complete
and the aligned-stack export run on the node too, and only their results
come back. The whole stack never crosses the wire.

### On Maxwell: `tktomo-track-maxwell`

All you need is an ssh alias for the login node (here `maxwell`) that logs
you in with a key, and `rsync` on the laptop. Then, from a checkout of this
repo with the `ui` extra installed:

```bash
tktomo-track-maxwell setup        # once: rsync the source and build a conda env
                                  # (conda-forge: tomopy, pyzmq, h5py, sklearn, ...)
                                  # under /gpfs/petra3/scratch/$USER/tktomo

tktomo-track-maxwell start /asap3/.../stack_preproc.h5
                                  # syncs the source, submits a SLURM job
                                  # (maxcpu, 4 h, 8 cpus by default), waits for
                                  # the node, opens the tunnel, starts the window;
                                  # closing the window cancels the job

tktomo-track-maxwell status       # your server jobs and the tunnel
tktomo-track-maxwell stop         # cancel them and close the tunnel
```

Options worth knowing: `--host` (another ssh alias), `--partition`,
`--time`, `--cpus`, `--mem`, `--port`, `--keep` (leave the job running
when the window closes, reconnect later with `start --no-sync`), `--no-app`
(just the tunnel, print the connect line), `--exact-frames`. The stack path
is optional; File > Open remote stack… asks for one on the node. Job logs
land in `<remote-dir>/jobs/slurm-<id>.out` on the cluster. The default
`--remote-dir` is Maxwell's per-user scratch (no quota, cleaned after three
months, and `setup` rebuilds it in minutes); a home directory's quota is
usually too small for the env.

Why it looks like this: a compute node cannot ssh out, so the laptop
tunnels *in* through the login node once SLURM has said which node the job
got, and the same key that opened the login node is offered to the compute
node. The server binds the node's loopback only. The launcher keeps
watching the tunnel while the window runs: if the link drops (a VPN
reconnect, say) the ssh keepalives end it within about 90 seconds and
the launcher reopens it, retrying every 20 seconds until the network is
back. The job and the window carry on meanwhile, so the first view change
after that simply works again. The read-ahead prefetcher pauses for
30 seconds after a run of failed fetches and resumes by itself.

### By hand, on any machine

```bash
# On the node (h5py and tomopy; scikit-learn and joblib for auto-complete;
# no Qt needed). The path is optional: File > Open remote stack… also works.
tktomo-track-server /beegfs/.../stack_preproc.h5

# On your laptop, through an SSH tunnel to that node:
ssh -N -L 5611:localhost:5611 <node>
python -m tktomo.ui.track_model_app --connect tcp://127.0.0.1:5611
```

With `--connect`, paths name files on the node: "Open remote stack…" asks
for one by text, the HDF5 dataset browser lists the remote file, and
"Export aligned stack" asks for an output path on the node and writes
there. Model and shift exports and session files are small and stay
local; a session records the endpoint it was labelled against, and loads
against the same server without re-reading the stack when it is still
open there.

> There is no authentication or encryption. Bind to `127.0.0.1` (the
> default) and use an SSH tunnel; never expose the port.

### What it does to keep view changes quick

A tunnelled link is latency-cheap and bandwidth-poor. Measured from a
laptop to a DESY node: 4 ms round trip, 0.4 to 0.6 MB/s, so a 1.3 MB
float32 frame took about two seconds and the transfer was the whole cost
of changing view. Two things narrow that:

- **Frames are packed to display precision.** Each one is quantised to 16
  bits over its own min/max and zlib'd, which is a little over 2x, and
  costs about 20 ms of node CPU. The error is bounded by 1/131070 of the
  frame's range, far below anything a display or an eye resolves, and
  frames are only ever displayed: auto-complete, the recon slice and the
  aligned export all run on the node, on the real pixels. A frame holding
  NaN or inf goes verbatim, and `--exact-frames` turns the packing off
  altogether.
- **The next few views are fetched while you work.** A background thread
  reads ahead in the direction you are moving, in the stride you are
  moving (the advance box means a session may step five views at a time),
  so a labelling pass down the stack mostly finds each frame already
  there. It fetches one frame at a time and re-plans after each, so a
  view you ask for never waits behind more than one prefetch.

A revisited view costs nothing either way: the client keeps the last 64 MB
of frames.

## Exports

- **Model + astra vectors**: the full fit in one HDF5 (coefficients,
  shifts, rotations, feature positions, labels, masks, provenance), plus
  a `(n_views, 12)` `parallel3d_vec` dataset whose ASTRA forward
  projection reproduces the fitted model exactly, tilts and per-view
  rotations included, for a geometry-aware GPU reconstruction.
- **slogger shifts.h5**: `sy`/`sx` and center/tilt attrs in the pipeline's
  convention (`sx = -dx/b`, `center = (c_raw - u0 - (b-1)/2)/b`), directly
  consumable by the graphite-ball pipeline's recon stage. The per-view
  rotations are written as `rot_*_rad` datasets for provenance only:
  `sx`/`sy` do not contain them, and an attribute says so.
- **Aligned projection stack**: undoes `dx`/`dy`, folds the `c(theta)`
  drift into per-view shifts and derotates the constant in-plane tilt
  plus the per-view rot beam (about the axis column, with the shift that
  makes an image-centre rotation act there), one affine resample per
  view. `beta`, any angle-dependence of `alpha`, rot axis and rot horiz
  cannot be expressed as 2D image transforms; they stay in the metadata.
- **Session**: labels, model, masks and UI state in a small HDF5 next to
  the data; the stack itself is re-read from its source on load.

## Recon slice

The Recon slice tab (top panel, next to the projection view)
reconstructs one detector row with tomopy gridrec at the
fitted center, after applying each view's full 2D correction: the dx/dy
shifts, the c(theta) drift, and derotation by the in-plane tilt
alpha(theta) plus the per-view rot beam, so the slice responds to the
tilt parameters. Rot axis goes to gridrec as a per-view angle increment.
Only beta and rot horiz stay out (they need 3D geometry, in 2D nothing
can honor them). On a remote stack the server must be at least as new as
the window for the angle increment to cross the link. A bin
selector mean-pools the slab before reconstruction (shifts, center and
row rescaled accordingly); cost falls roughly as bin cubed, so bin 2 or 4
makes live evaluation fluid and bin 1 is for the final look. Off by default; the
"Live" checkbox recomputes it (debounced) after every change. All
reconstruction runs on a single worker thread with single-flight
scheduling: tomopy segfaults when called from two threads at once, so a
request arriving while one runs replaces the pending one instead of
queueing.
