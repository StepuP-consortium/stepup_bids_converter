import numpy as np

import matplotlib.pyplot as plt

def plot_marker_events(marker_data_dict, show=True):
    """
    Plot marker events from a marker_data_dict.

    Parameters
    ----------
    marker_data_dict : dict
        Dictionary with marker_id as keys and dicts with 'data' and 'indices' as values.
    show : bool
        Whether to show the plot.
    """
    marker_ids = list(marker_data_dict.keys())
    n_markers = len(marker_ids)
    colors = plt.cm.tab10.colors  # Up to 10 distinct colors

    fig, ax = plt.subplots(figsize=(8, 2 + n_markers))
    handles, labels = [], []

    for idx, marker_id in enumerate(marker_ids):
        indices = marker_data_dict[marker_id]["indices"][0]
        y = np.full_like(indices, idx + 1)
        color = colors[idx % len(colors)]
        h = ax.plot(indices, y, ".", color=color, label=f"Marker {marker_id} (N={len(indices)})")[0]
        handles.append(h)
        labels.append(f"Marker {marker_id} (N={len(indices)})")

    ax.set_yticks(np.arange(1, n_markers + 1))
    ax.set_yticklabels(marker_ids)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Marker ID")
    ax.set_ylim(0, n_markers + 1)
    ax.grid(True)
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    plt.tight_layout()
    if show:
        plt.show()
    return fig
