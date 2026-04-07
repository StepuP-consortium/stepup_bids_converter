import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pyxdf
from collections import Counter

from motionbids import (
    MotionData,
    Channel,
    export_bids_motion,
    validate_motion_data,
    create_bids_directory_structure,
    export_dataset_description,
)

from src.config import DIR_BIDS_ROOT, DIR_PROJ
from src.plotting import plot_marker_events
from src.clustering_new import cluster_markers_temporal

# =========================================================================
# 1. Load XDF data
# =========================================================================
file_path = Path(r"C:\Users\juliu\Desktop\kiel\stepup_bids_converter\data\source\Bol14\T0")

xdf_files = list(file_path.glob("*WALKING_14*.xdf"))
if not xdf_files:
    raise FileNotFoundError("No XDF files found in the directory.")
for xdf_file in xdf_files:
    print(f"Found XDF file: {xdf_file.name}")

streams, fileheader = pyxdf.load_xdf(xdf_files[0], handle_clock_resets=False)
print("File loaded successfully.")

for i, stream in enumerate(streams):
    print(
        f"\nStream {i+1}: {stream['info']['name'][0]} "
        f"(type={stream['info']['type'][0]}, "
        f"channels={stream['info']['channel_count'][0]}, "
        f"srate={stream['info']['nominal_srate'][0]}, "
        f"n={len(stream['time_series'])})"
    )

# Parse filename metadata
fname = xdf_files[0].name
subject_id = fname.split("_")[2]
task = fname.split("_")[-1].split(".")[0]
session = fname.split("_")[3]

# =========================================================================
# 2. Extract Mocap stream
# =========================================================================
mocap_stream = [s for s in streams if s["info"]["type"][0] == "MoCap"][0]
mocap_times = mocap_stream["time_stamps"] - mocap_stream["time_stamps"][0]
mocap_raw = mocap_stream["time_series"]

SRATE_MOCAP = float(mocap_stream["info"]["nominal_srate"][0])
TRACKSYS = "qualisys"
N_MARKERS = 6

print(f"\nRaw data: {mocap_raw.shape[0]} samples, "
      f"{len(np.unique(mocap_raw[:, 3]))} unique marker IDs")

# =========================================================================
# 3. Cluster marker IDs (resolve ID jumps)
# =========================================================================
merged_data, id_mapping, cluster_info = cluster_markers_temporal(
    mocap_raw,
    mocap_times,
    n_markers=N_MARKERS,
    max_gap_seconds=2.0,              # hard cutoff at 1 second
    base_distance_threshold=30.0,     # mm, for near-instant jumps
    max_velocity_mmps=3000.0,         # mm/s, fast human motion
    spatial_consistency_threshold=100.0,  # mm, max median drift in cluster
)

print(f"\nAfter clustering: {Counter(merged_data[:, 3])}")

# =========================================================================
# 4. Select the N most common markers (the physical ones)
# =========================================================================
marker_ids_merged = merged_data[:, 3]
unique_ids, counts = np.unique(marker_ids_merged, return_counts=True)

# Sort by count descending, keep top N_MARKERS
sorted_markers = sorted(zip(unique_ids, counts), key=lambda x: x[1], reverse=True)

if len(sorted_markers) > N_MARKERS:
    print(f"\n⚠ {len(sorted_markers)} IDs remain, keeping top {N_MARKERS} by sample count.")
    print("  Discarded IDs (likely noise):")
    for mid, cnt in sorted_markers[N_MARKERS:]:
        print(f"    ID {int(mid)}: {cnt} samples")

selected_ids = [mid for mid, _ in sorted_markers[:N_MARKERS]]

# Build per-marker data dict with timestamps
marker_data_dict = {}
for marker_id in selected_ids:
    mask = marker_ids_merged == marker_id
    marker_data_dict[int(marker_id)] = {
        "data": merged_data[mask],
        "timestamps": mocap_times[mask],
        "indices": np.where(mask)[0],
    }

for marker_id, info in marker_data_dict.items():
    idx = info["indices"]
    print(
        f"Marker {marker_id}: {len(idx)} samples, "
        f"t=[{info['timestamps'][0]:.3f}, {info['timestamps'][-1]:.3f}]"
    )

# =========================================================================
# 5. Diagnostic plots
# =========================================================================
# Plot marker events (timeline)
fig = plot_marker_events(marker_data_dict, show=False)

# Plot 3D positions per marker
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axis_labels = ["X (mm)", "Y (mm)", "Z (mm)"]

for marker_id, info in marker_data_dict.items():
    t = info["timestamps"]
    pos = info["data"][:, :3]  # x, y, z
    for ax_idx in range(3):
        axes[ax_idx].plot(t, pos[:, ax_idx], label=f"ID {marker_id}", alpha=0.7)

for ax_idx, label in enumerate(axis_labels):
    axes[ax_idx].set_ylabel(label)
    axes[ax_idx].legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Time (s)")
axes[0].set_title("Marker positions over time")
plt.tight_layout()


plt.show()

# =========================================================================
# 6. Build time-aligned data array for BIDS
# =========================================================================
# Create a common time grid based on the nominal sample rate
t_min = mocap_times[0]
t_max = mocap_times[-1]
n_timepoints = int(np.round((t_max - t_min) * SRATE_MOCAP)) + 1
common_times = np.linspace(t_min, t_max, n_timepoints)

# Tolerance for matching a sample to a time grid point (half a sample period)
dt_tol = 0.5 / SRATE_MOCAP

n_channels = N_MARKERS * 3
motion_data = np.full((n_timepoints, n_channels), np.nan)

marker_names = []
for col_idx, (marker_id, info) in enumerate(marker_data_dict.items()):
    marker_name = f"marker{col_idx}"
    marker_names.append(marker_name)

    t_marker = info["timestamps"]
    pos = info["data"][:, :3]

    # For each marker sample, find the nearest grid point
    grid_indices = np.searchsorted(common_times, t_marker)
    # Clamp to valid range
    grid_indices = np.clip(grid_indices, 0, n_timepoints - 1)

    # Check if the nearest grid point is within tolerance
    dt_diff = np.abs(common_times[grid_indices] - t_marker)
    valid = dt_diff <= dt_tol

    valid_grid = grid_indices[valid]
    valid_pos = pos[valid]

    motion_data[valid_grid, col_idx * 3] = valid_pos[:, 0]      # x
    motion_data[valid_grid, col_idx * 3 + 1] = valid_pos[:, 1]  # y
    motion_data[valid_grid, col_idx * 3 + 2] = valid_pos[:, 2]  # z

# Report coverage
for i, name in enumerate(marker_names):
    valid_count = np.sum(~np.isnan(motion_data[:, i * 3]))
    pct = 100.0 * valid_count / n_timepoints
    print(f"  {name}: {valid_count}/{n_timepoints} samples ({pct:.1f}%)")

# =========================================================================
# 7. Create BIDS Channel objects
# =========================================================================
channels = [
    Channel(
        channel_name=f"{marker_names[i]}_{axis}",
        channel_component=axis,
        channel_type="POS",
        channel_tracked_point=marker_names[i],
        channel_units="mm",
    )
    for i in range(N_MARKERS)
    for axis in ["x", "y", "z"]
]

# =========================================================================
# 8. Create MotionData and export to BIDS
# =========================================================================
motion = MotionData(
    # Required fields
    subject_id=subject_id,
    task_name=task,
    tracksys=TRACKSYS,
    sampling_frequency=SRATE_MOCAP,
    tracked_points_count=N_MARKERS,
    # Recommended fields
    manufacturer="Qualisys",
    recording_type="continuous",
    motion_channel_count=n_channels,
    recording_duration=n_timepoints / SRATE_MOCAP,
    # Optional
    session_id=session if session else None,
    # Data
    data=motion_data,
    channels=channels,
)

try:
    validate_motion_data(motion)
    print("✓ Motion data is BIDS compliant!")
except Exception as e:
    print(f"✗ Validation warning: {e}")

# Create output directory
motion_dir = create_bids_directory_structure(
    base_dir=DIR_BIDS_ROOT,
    subject_id=subject_id,
    session_id=session if session else None,
)

# Export
files = export_bids_motion(
    motion,
    out_dir=motion_dir,
    validate=True,
    overwrite=True,
)

print(f"✓ Created BIDS files: {list(files.keys())}")
print(f"  Output directory: {motion_dir}")
