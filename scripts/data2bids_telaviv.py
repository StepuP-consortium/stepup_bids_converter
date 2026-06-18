"""Launcher: convert the Tel Aviv T0 dataset to BIDS.

All logic and configuration live in src/ (see src.bids_telaviv for the Tel Aviv
config / file routing and src.bids_convert for the shared conversion engine).
This script only invokes the conversion. Run from the repo so ``src`` imports:

    python scripts/data2bids_telaviv.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on sys.path

from src.bids_telaviv import main

if __name__ == "__main__":
    main()
