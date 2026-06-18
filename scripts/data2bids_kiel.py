"""Launcher: convert the full Kiel T0 dataset to BIDS.

All logic and configuration live in src/ (see src.bids_kiel for the Kiel config /
file routing and src.bids_convert for the shared conversion engine). This script
only invokes the conversion. Run from the repo so that ``src`` is importable:

    python scripts/data2bids_kiel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from src.bids_kiel import main

if __name__ == "__main__":
    main()
