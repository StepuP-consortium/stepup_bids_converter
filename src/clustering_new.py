import numpy as np
from collections import defaultdict


def cluster_markers_temporal(
    data,
    timestamps,
    n_markers=5,
    max_gap_seconds=1.0,
    base_distance_threshold=30.0,
    max_velocity_mmps=2000.0,
    spatial_consistency_threshold=80.0,
):
    """
    Cluster Qualisys marker IDs into physical markers by linking IDs that
    represent the same marker after an ID jump.

    Assumptions:
        - Once a marker ID disappears, it never comes back (Qualisys assigns
          a new ID).
        - Overlapping IDs can occur: during a jump, two IDs may briefly
          coexist in the same frame for the same physical point.
        - Data is sampled at ~100 Hz with human motion speeds.

    Parameters
    ----------
    data : ndarray, shape (n_samples, 4)
        Columns: [x, y, z, marker_id].
    timestamps : ndarray, shape (n_samples,)
        Timestamp (seconds) for each sample.
    n_markers : int
        Expected number of physical markers.
    max_gap_seconds : float
        Hard cutoff: never link IDs separated by more than this in time.
    base_distance_threshold : float
        Max distance (mm) to link IDs when the time gap is near zero.
    max_velocity_mmps : float
        Assumed max marker velocity (mm/s). The allowed linking distance
        grows linearly with the time gap: threshold = base + velocity * dt.
        For human motion, ~2000 mm/s covers fast hand movements.
    spatial_consistency_threshold : float
        Max median distance (mm) between overlapping time regions of two IDs
        for them to be considered the same physical marker.

    Returns
    -------
    new_data : ndarray, shape (n_samples, 4)
        Data with corrected marker IDs in column 3.
    id_mapping : dict
        {original_id: canonical_id}.
    cluster_info : dict
        {canonical_id: list of member IDs with metadata}.
    """

    marker_ids = data[:, 3]
    positions = data[:, :3]
    unique_ids = np.unique(marker_ids)

    # ------------------------------------------------------------------ #
    # STEP 1: Build per-ID summary (each ID is one contiguous block)
    # ------------------------------------------------------------------ #
    id_info = {}

    for mid in unique_ids:
        mask = marker_ids == mid
        indices = np.where(mask)[0]

        ts = timestamps[indices]
        pos = positions[indices]

        # Sort by time (should already be ordered, but be safe)
        order = np.argsort(ts)
        indices = indices[order]
        ts = ts[order]
        pos = pos[order]

        id_info[mid] = {
            "indices": indices,
            "timestamps": ts,
            "positions": pos,
            "t_start": ts[0],
            "t_end": ts[-1],
            "pos_first": pos[0],
            "pos_last": pos[-1],
            "pos_median": np.median(pos, axis=0),
            "n_samples": len(indices),
        }

    print("Marker ID summary:")
    for mid in sorted(unique_ids, key=lambda m: id_info[m]["t_start"]):
        info = id_info[mid]
        print(
            f"  ID {int(mid):5d}: t=[{info['t_start']:.3f}, {info['t_end']:.3f}], "
            f"n={info['n_samples']:6d}"
        )

    # ------------------------------------------------------------------ #
    # STEP 2: Compute adaptive distance threshold
    # ------------------------------------------------------------------ #
    def allowed_distance(dt):
        """
        Larger time gap -> allow more spatial distance (the marker moved).
        Linear model: base + velocity * dt, capped implicitly by max_gap.
        """
        return base_distance_threshold + max_velocity_mmps * dt

    # ------------------------------------------------------------------ #
    # STEP 3: Find candidate links between IDs
    # ------------------------------------------------------------------ #
    # Two types of links:
    #   A) Sequential: ID1 ends, ID2 starts shortly after (gap > 0)
    #   B) Overlapping: ID1 and ID2 overlap in time — check if they
    #      track the same point during the overlap

    candidates = []

    for id1 in unique_ids:
        info1 = id_info[id1]
        for id2 in unique_ids:
            if id1 >= id2:
                continue  # avoid duplicates and self
            info2 = id_info[id2]

            # Determine temporal relationship
            # Ensure id_a ends before/around id_b starts
            if info1["t_end"] <= info2["t_start"]:
                id_a, id_b = id1, id2
                info_a, info_b = info1, info2
            elif info2["t_end"] <= info1["t_start"]:
                id_a, id_b = id2, id1
                info_a, info_b = info2, info1
            else:
                # Overlapping in time — check overlap case
                _check_overlap_link(
                    id1, id2, info1, info2,
                    spatial_consistency_threshold, candidates
                )
                continue

            # Sequential case: id_a ends, id_b starts
            dt = info_b["t_start"] - info_a["t_end"]
            if dt > max_gap_seconds:
                continue

            dist = np.linalg.norm(info_a["pos_last"] - info_b["pos_first"])
            threshold = allowed_distance(dt)

            if dist < threshold:
                candidates.append({
                    "id_a": id_a,
                    "id_b": id_b,
                    "type": "sequential",
                    "dt": dt,
                    "dist": dist,
                    "threshold": threshold,
                    "score": dist / threshold,  # lower = more confident
                })

    # Sort candidates by confidence (lowest score first)
    candidates.sort(key=lambda c: c["score"])

    print(f"\n{len(candidates)} candidate links found:")
    for c in candidates:
        print(
            f"  {int(c['id_a']):5d} -> {int(c['id_b']):5d} [{c['type']:10s}] "
            f"dt={c['dt']:.4f}s, dist={c['dist']:.1f}mm "
            f"(threshold={c['threshold']:.1f}mm, score={c['score']:.3f})"
        )

    # ------------------------------------------------------------------ #
    # STEP 4: Greedy merging with spatial consistency validation
    # ------------------------------------------------------------------ #
    # Instead of blind union-find, we merge greedily (best links first)
    # and validate that the full cluster remains spatially consistent.

    # Track which cluster each ID belongs to
    cluster_of = {mid: mid for mid in unique_ids}  # each ID starts as its own cluster
    cluster_members = {mid: {mid} for mid in unique_ids}

    def get_cluster_positions(cluster_id):
        """Get all positions from all IDs in a cluster."""
        members = cluster_members[cluster_id]
        all_pos = [id_info[m]["pos_median"] for m in members]
        return np.array(all_pos)

    def clusters_spatially_consistent(cid_a, cid_b):
        """
        Check that merging two clusters wouldn't create a spatially
        inconsistent group. Compare median positions of all members.
        """
        pos_a = get_cluster_positions(cid_a)
        pos_b = get_cluster_positions(cid_b)

        # Check all cross-distances between median positions
        for pa in pos_a:
            for pb in pos_b:
                if np.linalg.norm(pa - pb) > spatial_consistency_threshold:
                    return False
        return True

    merged_count = 0
    for c in candidates:
        id_a, id_b = c["id_a"], c["id_b"]
        cid_a = cluster_of[id_a]
        cid_b = cluster_of[id_b]

        if cid_a == cid_b:
            continue  # already in the same cluster

        # Validate spatial consistency before merging
        if not clusters_spatially_consistent(cid_a, cid_b):
            print(
                f"  REJECTED: {int(id_a)} + {int(id_b)} — "
                f"cluster medians too far apart"
            )
            continue

        # Merge smaller cluster into larger
        if len(cluster_members[cid_a]) < len(cluster_members[cid_b]):
            cid_a, cid_b = cid_b, cid_a

        for mid in cluster_members[cid_b]:
            cluster_of[mid] = cid_a
        cluster_members[cid_a] |= cluster_members[cid_b]
        del cluster_members[cid_b]
        merged_count += 1

    # ------------------------------------------------------------------ #
    # STEP 5: Assign canonical IDs and remap
    # ------------------------------------------------------------------ #
    id_mapping = {}
    cluster_info = {}

    print(f"\n{len(cluster_members)} clusters after merging ({merged_count} merges):")
    for ci, (root, members) in enumerate(
        sorted(cluster_members.items(),
               key=lambda x: id_info[min(x[1], key=lambda m: id_info[m]["t_start"])]["t_start"])
    ):
        # Canonical = the ID with the most samples
        counts = {mid: id_info[mid]["n_samples"] for mid in members}
        canonical_id = max(counts, key=counts.get)
        total = sum(counts.values())

        print(f"\n  Cluster {ci} -> canonical ID {int(canonical_id)} ({total} samples)")
        for mid in sorted(members, key=lambda m: id_info[m]["t_start"]):
            info = id_info[mid]
            marker_str = " (canonical)" if mid == canonical_id else ""
            print(
                f"    ID {int(mid):5d}: {info['n_samples']:6d} samples, "
                f"t=[{info['t_start']:.3f} - {info['t_end']:.3f}]{marker_str}"
            )
            id_mapping[mid] = canonical_id

        cluster_info[canonical_id] = {
            "members": sorted(members),
            "total_samples": total,
        }

    # Apply remapping
    new_data = data.copy()
    for old_id, new_id in id_mapping.items():
        if old_id != new_id:
            mask = marker_ids == old_id
            new_data[mask, 3] = new_id

    n_final = len(np.unique(new_data[:, 3]))
    print(f"\nResult: {n_final} unique IDs (expected {n_markers})")
    if n_final != n_markers:
        print(
            f"  ⚠ Mismatch. Try adjusting max_gap_seconds, "
            f"base_distance_threshold, or spatial_consistency_threshold."
        )

    return new_data, id_mapping, cluster_info


def _check_overlap_link(id1, id2, info1, info2, threshold, candidates):
    """
    Check if two temporally overlapping IDs track the same physical point.

    During a Qualisys ID jump, the system may briefly produce two
    reconstructions for the same marker. If positions match during the
    overlap, they're the same marker.
    """
    # Find overlapping time region
    t_overlap_start = max(info1["t_start"], info2["t_start"])
    t_overlap_end = min(info1["t_end"], info2["t_end"])

    if t_overlap_end <= t_overlap_start:
        return  # no actual overlap

    # Get samples from each ID within the overlap window
    mask1 = (info1["timestamps"] >= t_overlap_start) & (
        info1["timestamps"] <= t_overlap_end
    )
    mask2 = (info2["timestamps"] >= t_overlap_start) & (
        info2["timestamps"] <= t_overlap_end
    )

    pos1 = info1["positions"][mask1]
    pos2 = info2["positions"][mask2]

    if len(pos1) == 0 or len(pos2) == 0:
        return

    # For each sample in the shorter series, find the nearest in time
    # from the longer series and compute distance
    ts1 = info1["timestamps"][mask1]
    ts2 = info2["timestamps"][mask2]

    # Use nearest-neighbor in time
    dists = []
    for t, p in zip(ts1, pos1):
        idx = np.argmin(np.abs(ts2 - t))
        dists.append(np.linalg.norm(p - pos2[idx]))

    median_dist = np.median(dists)

    if median_dist < threshold:
        # Determine which ends first (that one gets absorbed)
        if info1["t_end"] <= info2["t_end"]:
            id_a, id_b = id1, id2
        else:
            id_a, id_b = id2, id1

        overlap_duration = t_overlap_end - t_overlap_start
        candidates.append({
            "id_a": id_a,
            "id_b": id_b,
            "type": "overlap",
            "dt": -overlap_duration,  # negative = overlap
            "dist": median_dist,
            "threshold": threshold,
            "score": median_dist / threshold,
        })