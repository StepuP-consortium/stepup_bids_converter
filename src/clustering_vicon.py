"""Marker clustering for the Tel Aviv Vicon mocap stream.

Background
----------
The Tel Aviv setup streams a Vicon system over LSL with the *same* row layout as
the Kiel Qualisys stream — one row per detected marker per frame as
``[x, y, z, trajectory_id]`` — and the *same* failure mode: markers fragment into
many short-lived trajectory ids, so the raw ids cannot identify a marker. But the
marker *set* is different, so :func:`clustering_tracked.cluster_markers_tracked`
(which assumes the Kiel layout of 1 static treadmill + 2 trunk markers + 2 feet)
does not apply here and will crash or mislabel.

Empirically (verified across Walking1/2/3 for sub-001) the Tel Aviv take has
**5 physical markers**:

* **1 lower-back marker** — sits ~440 mm above the floor, essentially static
  (3D std ~3-4 mm), cleanly separated from everything else in height. Easy to
  resolve.
* **4 foot markers** — left/right heel and toe, hugging the floor (median height
  ~15-20 mm), sweeping fore-aft during gait. These fragment heavily; typically
  only 3 of the 4 are visible in any given frame, so they must be *tracked*, not
  read off the ids.

Approach: split by height, then track, then label
--------------------------------------------------
1. Reconstruct frames on the ``sr`` Hz grid (shared with the Qualisys path).
2. Resolve the lower-back marker by per-frame nearest-to-anchor on the high-Z
   rows (robust to id changes; the marker barely moves).
3. Track the four near-floor markers with the shared :class:`_MobileTracker`
   (trajectory-id continuity pin + gated spatial nearest-neighbour).
4. Label the four foot tracks as left/right heel/toe *after* tracking, from
   spatial/temporal cues, **reporting a confidence for every decision** because
   four fast, mutually-close, fragmented foot markers cannot be labelled as
   reliably as the Kiel two-foot case:

   * group the four tracks into two feet by fore-aft trajectory correlation
     (heel and toe of one foot move in phase; the contralateral foot is out of
     phase);
   * assign side from the mediolateral position of each foot (a documented
     convention, not a guaranteed anatomical side);
   * assign heel vs toe within a foot from fore-aft order along the direction of
     swing progression (the toe leads).
5. Drop isolated trajectory outliers (reflections), shared with the Qualisys path.

Public API
----------
:func:`cluster_markers_vicon` is the only entry point; it mirrors the return
contract of :func:`clustering_tracked.cluster_markers_tracked`
(``(grid, names, info)``).
"""

from __future__ import annotations

import numpy as np

from .clustering_tracked import (
    ClusterParams,
    _build_frame_index,
    _group_rows_by_frame,
    _MobileTracker,
    _remove_trajectory_outliers,
)

# Output channel order. The lower-back marker is first (most reliable), then the
# four feet. Foot names are filled in after labelling.
LOWER_BACK = "lower_back"
FOOT_NAMES = ["left_heel", "left_toe", "right_heel", "right_toe"]
N_MARKERS = 5
N_FEET = 4

# Labelling-confidence thresholds (drive the warnings and the QC gate).
MIN_HEEL_TOE_SEP = 30.0        # mm; heel and toe sit along the foot, well apart
MIN_GROUP_CONF = 0.3           # min correlation margin for a confident grouping
MIN_HEELTOE_CONSISTENCY = 0.7  # min fraction of frames the toe leads the heel


# --------------------------------------------------------------------------- #
# Lower-back (static) marker
# --------------------------------------------------------------------------- #
def _resolve_static_marker(xyz, frame_of_row, n_frames, mask, max_anchor_dist):
    """Place the near-static lower-back marker on the frame grid.

    The marker barely moves, so its whole-take median is a good anchor: for each
    frame keep the high-Z row nearest that anchor, dropping rows farther than
    ``max_anchor_dist`` (stray re-IDs / reflections).

    Returns ``(grid (n_frames, 3) with NaN gaps, anchor (3,) or None)``.
    """
    grid = np.full((n_frames, 3), np.nan)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return grid, None
    anchor = np.median(xyz[idx], axis=0)
    nearest = {}  # frame -> (distance_to_anchor, xyz)
    for i in idx:
        f = frame_of_row[i]
        d = np.linalg.norm(xyz[i] - anchor)
        if f not in nearest or d < nearest[f][0]:
            nearest[f] = (d, xyz[i])
    for f, (d, point) in nearest.items():
        if d <= max_anchor_dist:
            grid[f] = point
    return grid, anchor


# --------------------------------------------------------------------------- #
# Foot labelling (after tracking)
# --------------------------------------------------------------------------- #
def _horizontal_axes(foot_grid):
    """Pick which horizontal column is anteroposterior (AP) vs mediolateral (ML).

    During gait a foot sweeps far fore-aft (step length) but little side-to-side,
    so the horizontal axis with the larger per-track temporal spread is AP.

    Returns ``(ap_axis, ml_axis)`` as column indices into the xyz triple (0 or 1).
    """
    spreads = []
    for axis in (0, 1):
        per_track = []
        for k in range(foot_grid.shape[1]):
            col = foot_grid[:, k, axis]
            col = col[~np.isnan(col)]
            if col.size > 1:
                per_track.append(col.std())
        spreads.append(np.mean(per_track) if per_track else 0.0)
    ap_axis = 0 if spreads[0] >= spreads[1] else 1
    return ap_axis, 1 - ap_axis


def _pair_correlation(a, b):
    """Pearson correlation of two tracks over frames where both are present.

    Returns ``(corr, n_overlap)``; ``corr`` is NaN when the overlap is too small
    or either track is constant over the overlap.
    """
    both = (~np.isnan(a)) & (~np.isnan(b))
    n = int(both.sum())
    if n < 10:
        return np.nan, n
    av, bv = a[both], b[both]
    if av.std() < 1e-6 or bv.std() < 1e-6:
        return np.nan, n
    return float(np.corrcoef(av, bv)[0, 1]), n


def _group_into_feet(foot_grid, ap_axis):
    """Group the four foot tracks into two feet (heel+toe pairs).

    Heel and toe of one foot move in phase along the AP axis; the contralateral
    foot is out of phase. Of the three ways to split four tracks into two pairs,
    pick the one with the highest summed within-pair AP correlation.

    Returns ``(feet, confidence)`` where ``feet`` is a list of two ``(i, j)``
    index pairs and ``confidence`` is the margin between the best and second-best
    pairing (>0; larger is more certain, NaN if correlations are unusable).
    """
    ap = foot_grid[:, :, ap_axis]  # (n_frames, 4)
    partitions = [
        [(0, 1), (2, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
    ]
    scores = []
    for part in partitions:
        cs = [_pair_correlation(ap[:, i], ap[:, j])[0] for i, j in part]
        scores.append(np.nan if any(np.isnan(c) for c in cs) else sum(cs))
    scores = np.array(scores, dtype=float)
    if np.all(np.isnan(scores)):
        # fall back to the canonical pairing; flag with NaN confidence
        return partitions[0], np.nan
    best = int(np.nanargmax(scores))
    ordered = np.sort(scores[~np.isnan(scores)])[::-1]
    confidence = float(ordered[0] - ordered[1]) if ordered.size >= 2 else float(ordered[0])
    return partitions[best], confidence


def _foot_median(foot_grid, idx, axis, params):
    """Median position of a track on one axis over stance frames (fallback: all)."""
    col = foot_grid[:, idx, :]
    present = ~np.isnan(col[:, 0])
    stance = present & (col[:, 2] < params.stance_z)
    valid = stance if stance.any() else present
    return float(np.median(col[valid, axis])) if valid.any() else np.nan


def _forward_sign(foot_grid, ap_axis, params):
    """Sign of the AP direction the feet travel during swing.

    During swing (foot elevated) the foot advances in the facing direction; on a
    treadmill that is the fast phase, so the dominant sign of the AP velocity over
    elevated frames gives 'forward'. Returns +1 or -1 (defaults to +1 if unknown).
    """
    vel_sum = 0.0
    for k in range(foot_grid.shape[1]):
        col = foot_grid[:, k, :]
        ap = col[:, ap_axis]
        z = col[:, 2]
        d = np.diff(ap)
        elevated = (z[1:] > params.stance_z) & (~np.isnan(d))
        if elevated.any():
            vel_sum += np.nansum(d[elevated])
    return 1.0 if vel_sum >= 0 else -1.0


def _label_feet_vicon(foot_grid, params):
    """Assign left/right heel/toe to the four foot tracks.

    Returns ``(order, names, label_info)`` where ``order`` reindexes the four
    tracks to ``[left_heel, left_toe, right_heel, right_toe]`` and ``label_info``
    carries the per-decision confidences and any warnings.
    """
    warnings = []
    ap_axis, ml_axis = _horizontal_axes(foot_grid)

    # 1. group the four tracks into two feet (heel+toe pairs)
    feet, group_conf = _group_into_feet(foot_grid, ap_axis)
    if np.isnan(group_conf):
        warnings.append(
            "Could not group the four foot tracks into two feet by motion "
            "correlation (insufficient overlap); used a fallback pairing. "
            "Heel/toe/side labels are unreliable - verify against the plot."
        )
    elif group_conf < 0.3:
        warnings.append(
            f"Foot-grouping is low-confidence (correlation margin {group_conf:.2f} "
            f"between the best and next-best heel/toe pairing). Verify the labels."
        )

    # 2. side: the foot with the smaller mediolateral median is labelled left.
    #    This is a *convention* (handedness of the lab frame is unknown), reported
    #    with a standardized separation so the caller can judge it.
    def foot_ml(pair):
        vals = [_foot_median(foot_grid, k, ml_axis, params) for k in pair]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else np.nan

    ml0, ml1 = foot_ml(feet[0]), foot_ml(feet[1])
    # pooled ML spread across all four tracks for a standardized separation
    ml_stds = []
    for k in range(foot_grid.shape[1]):
        col = foot_grid[~np.isnan(foot_grid[:, k, 0]), k, ml_axis]
        if col.size > 1:
            ml_stds.append(col.std())
    pooled = np.sqrt(np.mean(np.square(ml_stds))) if ml_stds else np.nan
    side_sep_mm = abs(ml0 - ml1)
    side_sep_d = side_sep_mm / pooled if pooled and not np.isnan(pooled) else np.nan
    # Smaller mediolateral median == left (a convention; lab-frame handedness is
    # unknown). NaN-safe: if either foot lacks mediolateral data the order is
    # arbitrary, so keep the input order and flag the sides as unreliable.
    if np.isnan(ml0) or np.isnan(ml1):
        left_pair, right_pair = feet[0], feet[1]
        side_undetermined = True
    else:
        left_pair, right_pair = (feet[0], feet[1]) if ml0 <= ml1 else (feet[1], feet[0])
        side_undetermined = False
    side_low_conf = (side_undetermined or np.isnan(side_sep_d)
                     or side_sep_d < params.label_min_sep_d)
    if side_low_conf:
        warnings.append(
            f"Left/right side labels are low-confidence (mediolateral separation "
            f"{side_sep_mm:.0f} mm, d={side_sep_d:.2f}). Sides are a consistent "
            f"convention, not a proven anatomical side - confirm independently."
        )

    # 3. heel vs toe within each foot: the toe leads in the forward direction.
    fwd = _forward_sign(foot_grid, ap_axis, params)
    heeltoe_conf = {}

    def split_heel_toe(pair, side):
        ap0 = _foot_median(foot_grid, pair[0], ap_axis, params)
        ap1 = _foot_median(foot_grid, pair[1], ap_axis, params)
        # forward-most marker is the toe (NaN-safe: fall back to input order)
        if np.isnan(ap0) or np.isnan(ap1) or fwd * ap0 >= fwd * ap1:
            toe, heel = pair[0], pair[1]
        else:
            toe, heel = pair[1], pair[0]
        # confidence: fraction of co-present frames where the toe really leads,
        # and the AP separation between heel and toe (they sit along the foot, so
        # a near-zero separation means the two tracks may be the same point)
        a = foot_grid[:, toe, ap_axis]
        b = foot_grid[:, heel, ap_axis]
        both = (~np.isnan(a)) & (~np.isnan(b))
        consistency = float(np.mean(fwd * (a[both] - b[both]) > 0)) if both.any() else np.nan
        sep_mm = abs(ap0 - ap1)
        heeltoe_conf[side] = {"consistency": consistency, "ap_separation_mm": sep_mm}
        if np.isnan(consistency) or consistency < MIN_HEELTOE_CONSISTENCY:
            warnings.append(
                f"{side} heel/toe assignment is low-confidence "
                f"(toe-leads-heel in only {consistency:.0%} of shared frames). "
                f"Heel and toe may be swapped - verify."
            )
        if np.isnan(sep_mm) or sep_mm < MIN_HEEL_TOE_SEP:
            warnings.append(
                f"{side} heel and toe are only {sep_mm:.0f} mm apart along the foot "
                f"(expected well over {MIN_HEEL_TOE_SEP:.0f} mm); the two tracks may "
                f"be tracking the same point - heel/toe split is unreliable."
            )
        return heel, toe

    l_heel, l_toe = split_heel_toe(left_pair, "left")
    r_heel, r_toe = split_heel_toe(right_pair, "right")

    order = [l_heel, l_toe, r_heel, r_toe]
    label_info = {
        "ap_axis": {0: "X", 1: "Y"}[ap_axis],
        "ml_axis": {0: "X", 1: "Y"}[ml_axis],
        "group_confidence": group_conf,
        "side_separation_mm": side_sep_mm,
        "side_separation_d": side_sep_d,
        "side_low_confidence": bool(side_low_conf),
        "heeltoe_confidence": heeltoe_conf,
        "forward_sign": fwd,
    }
    return order, FOOT_NAMES, label_info, warnings


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #
def _quality_checks(grid, params):
    """Per-channel coverage (%) and maximum consecutive-frame jump (mm)."""
    coverage, max_jump = {}, {}
    frames_jump_gt_250 = 0
    for c in range(grid.shape[1]):
        col = grid[:, c, :]
        present = ~np.isnan(col[:, 0])
        coverage[c] = float(100.0 * np.mean(present))
        consecutive = present[:-1] & present[1:]
        if consecutive.any():
            jumps = np.linalg.norm(np.diff(col, axis=0), axis=1)[consecutive]
            max_jump[c] = float(jumps.max())
            frames_jump_gt_250 += int((jumps > 250).sum())
        else:
            max_jump[c] = np.nan
    return coverage, max_jump, frames_jump_gt_250


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def cluster_markers_vicon(rows, timestamps, sr=100.0, back_z_min=350.0, **overrides):
    """Cluster the Tel Aviv Vicon rows into 5 tracks: lower_back + 4 foot markers.

    Parameters
    ----------
    rows : ndarray, shape (n_samples, 4)
        Columns ``[x, y, z, trajectory_id]`` (the Vicon ``time_series``).
    timestamps : ndarray, shape (n_samples,)
        Per-row timestamps in seconds.
    sr : float
        Nominal sampling rate of the mocap stream (defines the frame grid).
    back_z_min : float
        Height (mm) above which a row is treated as the lower-back marker rather
        than a foot. The lower-back marker sits ~440 mm up and the feet stay below
        ~300 mm, so the default 350 mm splits them cleanly.
    **overrides
        Any field of :class:`clustering_tracked.ClusterParams` to override (these
        tune the shared foot tracker / outlier rejection).

    Returns
    -------
    grid : ndarray, shape (n_frames, 5, 3)
        Per-marker xyz on the frame grid, NaN where missing. Channel order is
        ``[lower_back, left_heel, left_toe, right_heel, right_toe]``.
    names : list[str]
        Tracked-point name per channel, aligned with ``grid``'s second axis.
    info : dict
        Diagnostics and QC, including ``names``, ``coverage`` (per channel %),
        ``max_jump`` (per channel mm), ``label`` (the labelling confidences),
        ``checks`` (pass/fail flags), ``all_checks_pass``, ``warnings`` and
        ``n_foot_candidates_max`` (the most simultaneous foot rows seen).

    Raises
    ------
    ValueError
        If the stream contains no finite rows at all.
    """
    params = ClusterParams(**overrides)

    rows = np.asarray(rows)
    if rows.size == 0:
        raise ValueError("Empty mocap stream: no rows to cluster.")
    xyz = rows[:, :3].astype(float)
    ids = rows[:, 3].astype(int)
    timestamps = np.asarray(timestamps, dtype=float)

    finite = np.isfinite(xyz).all(axis=1)
    if not finite.any():
        raise ValueError("Mocap stream has no finite coordinates.")
    if not finite.all():
        xyz, ids, timestamps = xyz[finite], ids[finite], timestamps[finite]

    frame_of_row, n_frames = _build_frame_index(timestamps, sr)
    warnings = []

    # 1. lower-back marker: the near-static high-Z rows.
    back_mask = xyz[:, 2] >= back_z_min
    back_grid, back_anchor = _resolve_static_marker(
        xyz, frame_of_row, n_frames, back_mask, params.rigid_max_anchor_dist)
    if back_anchor is None:
        warnings.append(
            f"No marker found above {back_z_min:.0f} mm; the lower-back channel "
            f"will be all-missing. Check back_z_min against the data."
        )

    # 2. foot markers: the near-floor rows, tracked with the shared tracker.
    foot_mask = (~back_mask) & (xyz[:, 2] < params.mobile_z_max)
    n_foot_max = 0
    if foot_mask.any():
        # most simultaneous foot rows in any frame (sanity: should reach 4)
        fcounts = np.bincount(frame_of_row[foot_mask], minlength=n_frames)
        n_foot_max = int(fcounts.max())
    frame_points, frame_ids = _group_rows_by_frame(
        xyz, ids, frame_of_row, n_frames, foot_mask)
    foot_grid, foot_track_ids = _MobileTracker(N_FEET, params).run(
        frame_points, frame_ids, n_frames)

    seeded = np.any(~np.isnan(foot_grid))
    if not seeded:
        warnings.append(
            "Could not seed four foot tracks (never four mutually-separated "
            "foot markers in one frame - e.g. a non-gait trial). Foot channels "
            "will be all-missing."
        )

    # 3. label the four foot tracks left/right heel/toe.
    if seeded:
        order, foot_names, label_info, label_warnings = _label_feet_vicon(foot_grid, params)
        warnings.extend(label_warnings)
        foot_grid = foot_grid[:, order, :]
    else:
        foot_names = list(FOOT_NAMES)
        label_info = None

    # 4. assemble the output grid: lower_back first, then the four feet.
    grid = np.full((n_frames, N_MARKERS, 3), np.nan)
    grid[:, 0, :] = back_grid
    grid[:, 1:, :] = foot_grid
    names = [LOWER_BACK] + foot_names

    # 5. drop isolated trajectory outliers (reflections etc.).
    outliers = _remove_trajectory_outliers(
        grid, params.outlier_spike_mm, params.outlier_max_gap, params.outlier_iters)

    # 6. quality control. The per-frame reflection dedup already keeps the four
    # tracks >= min_marker_sep apart, and the two feet legitimately cross during
    # gait, so a foot-to-foot distance gate would be either vacuous or misfire.
    # The meaningful signal is whether the labelling is confident: a trustworthy
    # result needs a clean foot grouping, a clear side separation and a reliable
    # heel/toe split on both feet (these also drive the warnings above).
    coverage, max_jump, frames_jump_gt_250 = _quality_checks(grid, params)

    def _labelled_confidently(li):
        if li is None:
            return False
        if li["side_low_confidence"]:
            return False
        if np.isnan(li["group_confidence"]) or li["group_confidence"] < MIN_GROUP_CONF:
            return False
        for v in li["heeltoe_confidence"].values():
            if np.isnan(v["consistency"]) or v["consistency"] < MIN_HEELTOE_CONSISTENCY:
                return False
            if np.isnan(v["ap_separation_mm"]) or v["ap_separation_mm"] < MIN_HEEL_TOE_SEP:
                return False
        return True

    checks = {
        "lower_back_present": coverage[0] > 50.0,
        "four_feet_seeded": bool(seeded),
        "feet_labelled_confidently": _labelled_confidently(label_info),
        "smooth": frames_jump_gt_250 == 0,
    }

    info = {
        "names": names,
        "coverage": coverage,
        "max_jump": max_jump,
        "frames_jump_gt_250mm": frames_jump_gt_250,
        "outliers_removed": {names[c]: n for c, n in outliers.items()},
        "outliers_removed_total": int(sum(outliers.values())),
        "n_foot_candidates_max": n_foot_max,
        "label": label_info,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "warnings": warnings,
    }
    return grid, names, info
