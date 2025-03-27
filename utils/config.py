import os
from pathlib import Path


def define_dir(root, *names):
    """define_dir create path handle and creates dir if not existend.

    Args:
        root (str): root to directory of interest
        names (tuple): subdirectories as separate arguments

    Returns:
        pathlib.Path: Pathlib handle of dir
    """
    path = Path(root)
    for name in names:
        path = path / name 
    path.mkdir(parents=True, exist_ok=True)
    return path

# If the user is not known, assume that the project is in the current working directory
dir_proj = Path.cwd().parent


###############################################################################
# These are relevant directories which are used in the analysis.

# (import) helper functions
dir_logs = define_dir(dir_proj, "logs")
dir_bids_root = define_dir(dir_proj, "data", "bids")
