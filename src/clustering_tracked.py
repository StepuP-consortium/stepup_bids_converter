"""Robust marker clustering for fragmented Qualisys mocap streams.

Background
----------
The Qualisys 6D stream stores one *row* per detected marker per frame as
``[x, y, z, trajectory_id]``. When a marker moves fast or is briefly occluded
(the feet during treadmill walking) Qualisys drops it and re-acquires it under a
brand-new ``trajectory_id``. In the Kiel recordings this fragments the two foot
markers into ~1000 short-lived ids, while the three slow / static markers
(treadmill, lower back, sternum) keep a single stable id for the whole take.

An earlier algorithm tried to re-link ids by pairwise median-position distance.
That fails for the feet: each fragment's median lands at a different point along
the gait sweep, so a spatial-consistency gate rejects almost every foot merge.
The feet never coalesce and the downstream "keep the top-N ids by sample count"
step then discards ~93 % of the foot data.

Approach here: track, then label
--------------------------------
1. Reconstruct frames by grouping rows onto the ``round(t * sr)`` grid.
2. Resolve the *rigid* markers directly from their stable ids, with per-frame
   ghost rejection (keep the row nearest that marker's anchor).
3. Pool the remaining near-ground rows as *mobile* markers (the feet) and track
   them with a two-stage per-frame matcher (id-continuity pin + gated spatial
   nearest-neighbour). See :class:`_MobileTracker`.
4. Label the recovered foot tracks left / right *after* tracking, from their
   stance-phase mediolateral position, reporting a confidence so the caller
   never silently claims anatomical sides it cannot prove.
5. Drop isolated samples that fit no trajectory (e.g. sunlight reflections).

Why continuity, not geometry, separates the feet
-------------------------------------------------
The two feet overlap on *every* individual axis: X is the fore-aft sweep (both
feet traverse it, out of phase), Z is vertical (both lift), and the mediolateral
(Y) separation is only tens of mm with the feet swaying across it. No static
spatial rule can tell them apart; only temporal continuity can, which is exactly
what the tracker uses.

Public API
----------
:func:`cluster_markers_tracked` is the only entry point. It returns
``(grid, names, info)`` — see its docstring for the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class ClusterParams:
    """Tunable parameters. Distances are in mm; per-frame values are at ``sr`` Hz.

    Defaults are tuned for the Kiel treadmill setup (5 markers at 100 Hz) and
    validated against real recordings; override any field via keyword arguments
    to :func:`cluster_markers_tracked`.
    """

    # --- marker classification ---
    stable_frac: float = 0.5
    """An id present in more than this fraction of frames is a rigid marker."""

    # --- rigid-marker resolution ---
    rigid_max_anchor_dist: float = 300.0
    """Reject a rigid sample farther than this from its marker's anchor (guards
    against a spurious detection that happened to carry the rigid id)."""

    # --- mobile-marker (foot) tracking ---
    mobile_z_max: float = 400.0
    """Mobile markers (feet) hug the ground; rows above this are stray re-IDs."""
    min_marker_sep: float = 80.0
    """Two physical mobile markers are never closer than this in 3D; a closer
    pair is a reflection ghost and is de-duplicated."""
    gate: float = 120.0
    """Base per-frame spatial gate for the matcher. A candidate farther than the
    gate from a track's prediction is rejected (artifact rejection)."""
    gate_widen_max: float = 4.0
    """Cap on how much the gate may grow during an occlusion, so a long gap can
    never widen it enough to admit an arbitrary nearby reflection."""
    coast_max: int = 12
    """Frames a track may coast on prediction before its velocity is zeroed."""
    vel_alpha: float = 0.5
    """Alpha-beta smoothing factor for the per-track velocity estimate."""

    # --- left / right labelling ---
    stance_z: float = 40.0
    """A mobile sample below this height counts as stance (used for labelling)."""
    label_min_sep_d: float = 1.0
    """Minimum standardized mediolateral separation (pooled-std units) for the
    left/right labels to be reported as confident."""

    # --- trajectory outlier rejection (e.g. sunlight reflections) ---
    outlier_spike_mm: float = 50.0
    """A sample deviating from the straight line between its temporal neighbours
    by more than this is dropped. Safe to keep small: a real marker's chord
    residual is only ~mm even at full swing speed (~22 mm worst case after a
    3-frame gap), whereas a flickering reflection deviates by hundreds."""
    outlier_max_gap: int = 3
    """Only spike-test a sample whose bracketing present neighbours are within
    this many frames (beyond that the local chord is an unreliable model)."""
    outlier_iters: int = 3
    """Repeat the spike pass this many times to peel short artifact bursts from
    the edges inward."""


_BIG = 1e9  # sentinel cost for gated / forbidden assignment cells


# --------------------------------------------------------------------------- #
# Frame grid + small geometry helpers
# --------------------------------------------------------------------------- #
def _build_frame_index(timestamps, sr):
    """Map each row to a frame on the ``sr`` Hz grid.

    Returns
    -------
    frame_of_row : (n_rows,) int array, the 0-based frame index of each row.
    n_frames : int, total number of frames spanning the recording.
    """
    t = np.asarray(timestamps, dtype=float)
    t = t - t[0]
    frame_of_row = np.round(t * sr).astype(int)
    frame_of_row -= frame_of_row.min()
    return frame_of_row, int(frame_of_row.max()) + 1


def _dedupe_reflections(points, ids, min_sep):
    """Drop reflection ghosts: greedily keep points >= ``min_sep`` (3D) apart."""
    if len(points) <= 1:
        return points, ids
    keep = [0]
    for i in range(1, len(points)):
        if all(np.linalg.norm(points[i] - points[k]) >= min_sep for k in keep):
            keep.append(i)
    keep = np.array(keep)
    return points[keep], ids[keep]


def _group_rows_by_frame(xyz, ids, frame_of_row, n_frames, mask):
    """Bucket the masked rows into per-frame ``(points, ids)`` lists."""
    points = [[] for _ in range(n_frames)]
    row_ids = [[] for _ in range(n_frames)]
    for i in np.where(mask)[0]:
        f = frame_of_row[i]
        points[f].append(xyz[i])
        row_ids[f].append(int(ids[i]))
    points = [np.array(a) if a else np.empty((0, 3)) for a in points]
    row_ids = [np.array(a, dtype=int) for a in row_ids]
    return points, row_ids


# --------------------------------------------------------------------------- #
# Step 1: classify trajectory ids
# --------------------------------------------------------------------------- #
def _classify_marker_ids(ids, n_frames, stable_frac):
    """Split trajectory ids into stable (rigid) ids and everything else.

    A rigid marker keeps one id for most of the take, so its id appears in more
    than ``stable_frac`` of the frames; fragmented (mobile) markers do not.
    """
    uniq, counts = np.unique(ids, return_counts=True)
    stable_ids = np.sort(uniq[counts > stable_frac * n_frames])
    return stable_ids


# --------------------------------------------------------------------------- #
# Step 2: rigid markers
# --------------------------------------------------------------------------- #
@dataclass
class RigidMarker:
    """Summary of one stable-id (rigid) marker."""

    trajectory_id: int
    median: np.ndarray  # (3,) whole-take median position == its anchor
    std: float          # 3D position spread; ~0 for the static treadmill marker


def _resolve_rigid_markers(xyz, ids, frame_of_row, n_frames, stable_ids,
                           max_anchor_dist):
    """Place each rigid marker on the frame grid, rejecting ghosts and artifacts.

    For each frame the row nearest the marker's anchor (its whole-take median) is
    kept; duplicate / reflection rows carrying the same id are dropped. A frame
    is left empty (NaN) when even the nearest candidate is farther than
    ``max_anchor_dist`` from the anchor.

    Returns
    -------
    grid : (n_frames, n_stable, 3) array, NaN where the id is absent.
    markers : list[RigidMarker], one per stable id (same order as ``stable_ids``).
    """
    markers = []
    for sid in stable_ids:
        p = xyz[ids == sid]
        markers.append(RigidMarker(
            trajectory_id=int(sid),
            median=np.median(p, axis=0),
            std=float(np.linalg.norm(p.std(axis=0))),
        ))

    grid = np.full((n_frames, len(markers), 3), np.nan)
    for slot, marker in enumerate(markers):
        nearest = {}  # frame -> (distance_to_anchor, xyz)
        for i in np.where(ids == marker.trajectory_id)[0]:
            f = frame_of_row[i]
            d = np.linalg.norm(xyz[i] - marker.median)
            if f not in nearest or d < nearest[f][0]:
                nearest[f] = (d, xyz[i])
        for f, (d, point) in nearest.items():
            if d <= max_anchor_dist:
                grid[f, slot] = point
    return grid, markers


def _order_and_name_rigid_markers(markers):
    """Order rigid slots and assign tracked-point names.

    The treadmill marker is the one with the smallest position spread (it is
    static); the remaining trunk markers are ordered low -> high by median height.
    For the Kiel two-trunk layout that is ``lower_back`` then ``sternum``.

    Returns
    -------
    order : list[int], indices into ``markers`` in output order.
    names : list[str], the tracked-point name for each ordered slot.
    """
    if not markers:
        return [], []
    treadmill = int(np.argmin([m.std for m in markers]))
    trunk = sorted((i for i in range(len(markers)) if i != treadmill),
                   key=lambda i: markers[i].median[2])
    order = [treadmill] + trunk
    trunk_names = (["lower_back", "sternum"] if len(trunk) == 2
                   else [f"trunk_{j}" for j in range(len(trunk))])
    return order, ["treadmill"] + trunk_names


# --------------------------------------------------------------------------- #
# Step 3: mobile-marker (foot) tracking
# --------------------------------------------------------------------------- #
@dataclass
class _Track:
    """Mutable state of one tracked mobile marker."""

    pos: Optional[np.ndarray] = None              # last accepted position
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    miss: int = 0                                 # consecutive unmatched frames
    current_id: Optional[int] = None              # trajectory id held last frame

    @property
    def prediction(self):
        """Constant-velocity prediction for the next frame."""
        return self.pos + self.vel


class _MobileTracker:
    """Track ``n_tracks`` fragmented markers with two-stage data association.

    For each frame the candidate points are matched to the tracks in two stages:

    * **Stage A - id-continuity pin.** Within a single fragment the Qualisys
      trajectory id is constant, so a candidate carrying the id a track held last
      frame is matched to it immediately. This holds identity through the gait
      crossing, where the two feet overlap spatially and a pure nearest-neighbour
      rule would swap them.
    * **Stage B - gated nearest-neighbour.** The remaining tracks and points are
      matched by a Hungarian assignment on Euclidean distance to each track's
      constant-velocity prediction. A candidate outside every track's gate is
      left unmatched and thereby rejected as an artifact. The gate widens while a
      track is missing (capped by ``gate_widen_max``) to allow re-acquisition.

    Unmatched tracks coast on their prediction; after ``coast_max`` missed frames
    their velocity is zeroed so a long gap cannot extrapolate wildly.
    """

    def __init__(self, n_tracks, params):
        self.n = n_tracks
        self.p = params
        self.tracks = [_Track() for _ in range(n_tracks)]

    # -- per-frame candidate preparation -------------------------------------
    def _candidates(self, points, ids):
        """Reflection-deduplicated candidates for one frame.

        We deliberately do *not* truncate to the lowest-Z points: a floor-level
        reflection can sit below a real foot in swing and would evict it. Every
        deduplicated candidate is offered to the matcher, which keeps only those
        within reach of a track and rejects the rest.
        """
        return _dedupe_reflections(points, ids, self.p.min_marker_sep)

    # -- data association ----------------------------------------------------
    def _match(self, points, ids):
        """Return ``{track_index: point_index}`` for one frame's candidates."""
        assignment = {}
        used = set()

        # Stage A: hard trajectory-id continuity pin
        for k, track in enumerate(self.tracks):
            if track.current_id is None:
                continue
            for j in range(len(points)):
                if j not in used and ids[j] == track.current_id:
                    assignment[k] = j
                    used.add(j)
                    break

        # Stage B: gated spatial Hungarian on the leftovers
        free_k = [k for k in range(self.n) if k not in assignment]
        free_j = [j for j in range(len(points)) if j not in used]
        if free_k and free_j:
            cost = np.full((len(free_k), len(free_j)), _BIG)
            for a, k in enumerate(free_k):
                gate = self.p.gate * min(1 + self.tracks[k].miss,
                                         self.p.gate_widen_max)
                pred = self.tracks[k].prediction
                for b, j in enumerate(free_j):
                    d = np.linalg.norm(pred - points[j])
                    if d <= gate:
                        cost[a, b] = d
            # full matrix + sentinel cost => linear_sum_assignment never fails
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < _BIG:
                    assignment[free_k[r]] = free_j[c]
        return assignment

    # -- track update --------------------------------------------------------
    def _update(self, track, point, point_id):
        # smooth velocity only across a real consecutive step
        if track.miss == 0 and track.pos is not None:
            track.vel = (self.p.vel_alpha * (point - track.pos)
                         + (1 - self.p.vel_alpha) * track.vel)
        else:
            track.vel = np.zeros(3)
        track.pos = point.copy()
        track.current_id = int(point_id)
        track.miss = 0

    def _coast(self, track):
        track.miss += 1
        if track.miss > self.p.coast_max:
            track.vel = np.zeros(3)

    # -- main loop -----------------------------------------------------------
    def run(self, frame_points, frame_ids, n_frames):
        """Track across all frames.

        Returns
        -------
        grid : (n_frames, n_tracks, 3) array, NaN where a track is unobserved.
        track_ids : (n_frames, n_tracks) int, trajectory id assigned per frame
            (-1 where unobserved); used for the swap audit.
        """
        grid = np.full((n_frames, self.n, 3), np.nan)
        track_ids = np.full((n_frames, self.n), -1, dtype=int)

        init = self._initialise(frame_points, frame_ids, grid, track_ids)
        if init is None:
            return grid, track_ids  # no mobile data at all

        for f in range(init + 1, n_frames):
            points, ids = self._candidates(frame_points[f], frame_ids[f])
            if len(points) == 0:
                for track in self.tracks:
                    self._coast(track)
                continue
            assignment = self._match(points, ids)
            for k, track in enumerate(self.tracks):
                if k in assignment:
                    j = assignment[k]
                    self._update(track, points[j], ids[j])
                    grid[f, k] = track.pos
                    track_ids[f, k] = track.current_id
                else:
                    self._coast(track)
        return grid, track_ids

    def _initialise(self, frame_points, frame_ids, grid, track_ids):
        """Seed the tracks from the first frame that shows all of them.

        Mobile markers hug the ground, so the ``n_tracks`` lowest points are
        taken and seeded in ascending-X order (an arbitrary but stable initial
        labelling; final left/right naming happens after tracking).
        """
        for f in range(len(frame_points)):
            points, ids = self._candidates(frame_points[f], frame_ids[f])
            if len(points) >= self.n:
                lowest = np.argsort(points[:, 2])[:self.n]
                points, ids = points[lowest], ids[lowest]
                for k, j in enumerate(np.argsort(points[:, 0])[:self.n]):
                    self.tracks[k].pos = points[j].copy()
                    self.tracks[k].current_id = int(ids[j])
                    grid[f, k] = self.tracks[k].pos
                    track_ids[f, k] = self.tracks[k].current_id
                return f
        return None


# --------------------------------------------------------------------------- #
# Step 4: left / right labelling (after tracking)
# --------------------------------------------------------------------------- #
@dataclass
class LabelResult:
    """Outcome of labelling two mobile tracks as left / right feet."""

    order: list           # track indices reordered as [left, right]
    names: list           # tracked-point names aligned with ``order``
    separation_mm: float  # raw |median difference| on the chosen axis
    separation_d: float   # standardized separation (pooled-std units)
    axis: Optional[str]   # "X" / "Y" / None — the mediolateral axis used
    low_confidence: bool
    warning: Optional[str]


def _label_feet(mob_grid, params):
    """Assign left/right foot labels from stance-phase mediolateral position.

    The feet overlap on every axis, so the only meaningful side cue is the
    mediolateral one. We don't assume which stored axis that is: of the two
    horizontal axes (X, Y) we pick the one that best separates the tracks during
    stance (largest standardized median difference); the foot with the smaller
    median there is labelled left. Confidence is judged on the *standardized*
    separation, because the feet overlap in absolute mm yet can still be cleanly
    separated during stance (small within-foot variance).
    """
    n = mob_grid.shape[1]
    if n != 2:
        return LabelResult(order=list(range(n)),
                           names=[f"foot_{j}" for j in range(n)],
                           separation_mm=np.nan, separation_d=np.nan,
                           axis=None, low_confidence=False, warning=None)

    def stance_median_std(track, axis):
        col = mob_grid[:, track, :]
        stance = (~np.isnan(col[:, 0])) & (col[:, 2] < params.stance_z)
        valid = stance if stance.any() else ~np.isnan(col[:, 0])
        if not valid.any():
            return np.nan, np.nan
        return float(np.median(col[valid, axis])), float(np.std(col[valid, axis]))

    best = None  # (standardized_separation, axis, median0, median1)
    for axis in (0, 1):  # X, Y  (Z excluded: vertical is not a side cue)
        m0, s0 = stance_median_std(0, axis)
        m1, s1 = stance_median_std(1, axis)
        pooled = np.sqrt((s0 ** 2 + s1 ** 2) / 2.0) or np.nan
        d = abs(m0 - m1) / pooled if pooled and not np.isnan(pooled) else 0.0
        if best is None or d > best[0]:
            best = (d, axis, m0, m1)

    sep_d, axis, m0, m1 = best
    order = [0, 1] if m0 <= m1 else [1, 0]   # smaller median == left
    axis_name = {0: "X", 1: "Y"}[axis]
    low_conf = sep_d < params.label_min_sep_d
    warning = None
    if low_conf:
        warning = (
            f"Left/right foot labels are low-confidence (separation "
            f"{abs(m0 - m1):.0f} mm, d={sep_d:.2f} on the {axis_name} axis). They "
            f"are a consistent convention, not a guaranteed anatomical side; "
            f"confirm with an independent cue (e.g. heel-strike laterality)."
        )
    return LabelResult(order=order, names=["left_foot", "right_foot"],
                       separation_mm=float(abs(m0 - m1)), separation_d=float(sep_d),
                       axis=axis_name, low_confidence=low_conf, warning=warning)


# --------------------------------------------------------------------------- #
# Step 5: trajectory outlier rejection
# --------------------------------------------------------------------------- #
def _remove_trajectory_outliers(grid, spike_mm, max_gap, iters):
    """Drop isolated samples that fit no trajectory (e.g. sunlight reflections).

    A present sample bracketed by present neighbours within ``max_gap`` frames is
    compared against the straight line interpolated between those neighbours. If
    it deviates by more than ``spike_mm`` it is a spike that "leaves and returns"
    — impossible at this frame rate — and is set to NaN. The test is motion-aware
    (a genuinely fast but smooth move lies on the local chord, so it is kept) and
    repeated ``iters`` times to peel short bursts from the edges inward.

    Mutates ``grid`` in place. Returns ``{channel: n_samples_removed}``.
    """
    n_ch = grid.shape[1]
    removed = {c: 0 for c in range(n_ch)}
    for c in range(n_ch):
        for _ in range(iters):
            present = np.where(~np.isnan(grid[:, c, 0]))[0]
            if len(present) < 3:
                break
            drop = []
            for j in range(1, len(present) - 1):
                before, here, after = present[j - 1], present[j], present[j + 1]
                if (here - before) > max_gap or (after - here) > max_gap:
                    continue
                w = (here - before) / (after - before)
                chord = grid[before, c] + (grid[after, c] - grid[before, c]) * w
                if np.linalg.norm(grid[here, c] - chord) > spike_mm:
                    drop.append(here)
            if not drop:
                break
            grid[drop, c, :] = np.nan
            removed[c] += len(drop)
    return removed


# --------------------------------------------------------------------------- #
# Quality control
# --------------------------------------------------------------------------- #
def _count_id_cross_assignments(track_ids):
    """Count frames where a track adopts the id the *other* track held last.

    This is the true "no swap" metric: foot identity is pinned by trajectory id,
    so a genuine swap shows up as one track inheriting its neighbour's last id.
    """
    n_frames, n = track_ids.shape
    last = {k: None for k in range(n)}
    events = 0
    for f in range(n_frames):
        cur = track_ids[f]
        for k in range(n):
            if cur[k] < 0:
                continue
            for other in range(n):
                if other != k and last[other] is not None and cur[k] == last[other]:
                    events += 1
        for k in range(n):
            if cur[k] >= 0:
                last[k] = cur[k]
    return events


def _quality_checks(grid, mob_grid, mob_track_ids, n_rigid, n_markers,
                    n_mobile, params):
    """Compute coverage, per-channel max jump, and the QC pass/fail flags.

    Returns a dict with ``coverage``, ``max_jump``, ``checks`` and the supporting
    scalar metrics. Channel keys are integer column indices.
    """
    n_frames = grid.shape[0]
    coverage = {c: float(100.0 * np.mean(~np.isnan(grid[:, c, 0])))
                for c in range(grid.shape[1])}

    # per-channel maximum consecutive-frame jump (smoothness probe)
    max_jump = {}
    frames_jump_gt_250 = 0
    for c in range(grid.shape[1]):
        col = grid[:, c, :]
        present = ~np.isnan(col[:, 0])
        consecutive = present[:-1] & present[1:]
        if consecutive.any():
            jumps = np.linalg.norm(np.diff(col, axis=0), axis=1)[consecutive]
            max_jump[c] = float(jumps.max())
            frames_jump_gt_250 += int((jumps > 250).sum())
        else:
            max_jump[c] = np.nan

    checks = {"exactly_n_tracks": grid.shape[1] == n_markers}

    # the two mobile tracks must never collapse onto the same point
    min_mobile_distance = None
    if n_mobile >= 2:
        a, b = mob_grid[:, 0, :], mob_grid[:, 1, :]
        both = (~np.isnan(a[:, 0])) & (~np.isnan(b[:, 0]))
        dist = np.linalg.norm(a[both] - b[both], axis=1) if both.any() else np.array([np.inf])
        min_mobile_distance = float(dist.min())
        checks["no_track_collapse"] = bool(dist.min() >= params.min_marker_sep)

    swaps = _count_id_cross_assignments(mob_track_ids)
    checks["no_id_swaps"] = swaps == 0
    checks["smooth"] = frames_jump_gt_250 == 0
    checks["rigid_stable"] = all(
        np.isnan(max_jump[c]) or max_jump[c] <= params.gate
        for c in range(n_rigid)
    )

    return dict(coverage=coverage, max_jump=max_jump, checks=checks,
                min_mobile_distance=min_mobile_distance,
                id_cross_assignments=int(swaps),
                frames_jump_gt_250mm=frames_jump_gt_250)


def _label_high_markers(high_grid):
    """Order and name the high-band markers by role.

    The treadmill marker does not move, so it has the smallest 3D positional
    spread; of the remaining trunk markers the lower-median-height one is the
    lower back and the higher one is the sternum. Robust to which markers keep a
    stable id and which fragment (the tracking already happened).

    Returns ``(order, names)`` reindexing the high tracks to
    ``[treadmill, lower_back, sternum]`` (for the 3-marker Kiel case).
    """
    n = high_grid.shape[1]
    stds, med_z = [], []
    for k in range(n):
        col = high_grid[:, k, :]
        v = ~np.isnan(col[:, 0])
        stds.append(float(np.linalg.norm(col[v].std(axis=0))) if v.any() else np.inf)
        med_z.append(float(np.median(col[v, 2])) if v.any() else -np.inf)
    treadmill = int(np.argmin(stds))
    trunk = sorted((k for k in range(n) if k != treadmill), key=lambda k: med_z[k])
    order = [treadmill] + trunk
    names = (["treadmill", "lower_back", "sternum"] if n == 3
             else ["treadmill"] + [f"trunk_{j}" for j in range(len(trunk))])
    return order, names


def _dominant_ids(track_ids):
    """Most frequent trajectory id per track (ignoring -1); for diagnostics."""
    out = []
    for k in range(track_ids.shape[1]):
        col = track_ids[:, k]
        col = col[col >= 0]
        if col.size:
            vals, counts = np.unique(col, return_counts=True)
            out.append(int(vals[counts.argmax()]))
        else:
            out.append(-1)
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def cluster_markers_tracked(rows, timestamps, sr=100.0, n_markers=5, n_feet=2,
                            **overrides):
    """Cluster fragmented Qualisys marker rows into ``n_markers`` continuous tracks.

    Parameters
    ----------
    rows : ndarray, shape (n_samples, 4)
        Columns ``[x, y, z, trajectory_id]`` (the Qualisys 6D ``time_series``).
    timestamps : ndarray, shape (n_samples,)
        Per-row timestamps in seconds.
    sr : float
        Nominal sampling rate of the mocap stream (defines the frame grid).
    n_markers : int
        Total number of physical markers expected.
    **overrides
        Any field of :class:`ClusterParams` to override.

    n_feet : int
        Number of near-floor (foot) markers; the rest (``n_markers - n_feet``)
        are the high markers (treadmill + trunk). For Kiel: 2 feet, 3 high.
    **overrides
        Any field of :class:`ClusterParams` to override.

    Returns
    -------
    grid : ndarray, shape (n_frames, n_markers, 3)
        Per-marker xyz on the ``sr`` frame grid, NaN where a marker is missing.
        Channel order is the high markers first (treadmill, lower_back, sternum),
        then the feet (left_foot, right_foot).
    names : list[str]
        Tracked-point name per channel, aligned with ``grid``'s second axis.
    info : dict
        Diagnostics and QC. Keys: ``high_marker_ids``, ``foot_ids``, ``names``,
        ``coverage`` (per channel %), ``max_jump`` (per channel mm),
        ``min_mobile_distance``, ``id_cross_assignments``, ``frames_jump_gt_250mm``,
        ``outliers_removed`` (per channel) and ``outliers_removed_total``,
        ``label_confidence``, ``label_separation_d``, ``label_axis``,
        ``label_low_confidence``, ``checks`` (dict of pass/fail flags),
        ``all_checks_pass`` and ``warnings``.

    Notes
    -----
    Markers are separated by *height*, not by whether their trajectory id is
    stable. The treadmill + trunk markers sit well above the floor while the feet
    hug it, so a foot that happens to keep a stable id is still tracked as a foot
    (and a trunk marker that fragments is still tracked as a trunk marker). Both
    bands are tracked with the shared :class:`_MobileTracker` (id-continuity pin +
    gated nearest-neighbour), then role-labelled.
    """
    params = ClusterParams(**overrides)

    rows = np.asarray(rows)
    xyz = rows[:, :3].astype(float)
    ids = rows[:, 3].astype(int)
    timestamps = np.asarray(timestamps, dtype=float)

    # Drop non-finite rows (NaN/Inf coordinates from dropped frames or glitches).
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        xyz, ids, timestamps = xyz[finite], ids[finite], timestamps[finite]

    frame_of_row, n_frames = _build_frame_index(timestamps, sr)
    warnings = []
    n_high = n_markers - n_feet
    if n_high < 1 or n_feet < 0:
        raise ValueError(f"n_feet={n_feet} incompatible with n_markers={n_markers}")

    # 1. Split rows by height: feet near the floor, treadmill + trunk above it.
    low_mask = xyz[:, 2] < params.mobile_z_max
    high_mask = ~low_mask

    # 2. Track each band independently (handles stable AND fragmented ids alike).
    fp_h, fi_h = _group_rows_by_frame(xyz, ids, frame_of_row, n_frames, high_mask)
    high_grid, high_tids = _MobileTracker(n_high, params).run(fp_h, fi_h, n_frames)
    fp_l, fi_l = _group_rows_by_frame(xyz, ids, frame_of_row, n_frames, low_mask)
    foot_grid, foot_tids = _MobileTracker(n_feet, params).run(fp_l, fi_l, n_frames)

    # 3. Role-label: high markers by static/height, feet by mediolateral side.
    high_order, high_names = _label_high_markers(high_grid)
    high_grid = high_grid[:, high_order, :]
    high_tids = high_tids[:, high_order]
    label = _label_feet(foot_grid, params)
    if label.warning:
        warnings.append(label.warning)
    foot_grid = foot_grid[:, label.order, :]
    foot_tids = foot_tids[:, label.order]

    # 4. Assemble the output grid: high markers first, then feet.
    grid = np.full((n_frames, n_markers, 3), np.nan)
    grid[:, :n_high, :] = high_grid
    grid[:, n_high:, :] = foot_grid
    names = list(high_names) + list(label.names)

    # 5. Drop isolated trajectory outliers (sunlight reflections etc.).
    outliers = _remove_trajectory_outliers(
        grid, params.outlier_spike_mm, params.outlier_max_gap, params.outlier_iters)
    foot_grid = grid[:, n_high:, :]  # keep the QC view in sync with the cleaned grid

    # 6. Quality control + assemble the info report.
    qc = _quality_checks(grid, foot_grid, foot_tids, n_high, n_markers, n_feet, params)
    info = {
        "high_marker_ids": _dominant_ids(high_tids),
        "foot_ids": _dominant_ids(foot_tids),
        "names": names,
        "coverage": qc["coverage"],
        "max_jump": qc["max_jump"],
        "min_mobile_distance": qc["min_mobile_distance"],
        "id_cross_assignments": qc["id_cross_assignments"],
        "frames_jump_gt_250mm": qc["frames_jump_gt_250mm"],
        "outliers_removed": {names[c]: n for c, n in outliers.items()},
        "outliers_removed_total": int(sum(outliers.values())),
        "label_confidence": label.separation_mm,
        "label_separation_d": label.separation_d,
        "label_axis": label.axis,
        "label_low_confidence": bool(label.low_confidence),
        "checks": qc["checks"],
        "all_checks_pass": all(qc["checks"].values()),
        "warnings": warnings,
    }
    return grid, names, info
