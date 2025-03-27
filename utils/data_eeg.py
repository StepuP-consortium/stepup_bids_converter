import datetime
import mne
from mnelab.io.xdf import read_raw_xdf
import numpy as np
from pathlib import Path
import pandas as pd
import pyxdf

def get_info_rs_eeg_files(data_dir):
    """
    Processes EEG (.vhdr) files in a directory, storing information
    in a growing DataFrame and skipping previously processed files.

    Args:
        data_dir (Path): Path to the directory containing EEG files.

    Returns:
        pandas.DataFrame: The growing DataFrame containing information from processed files.
    """

    processed_files_path = data_dir / 'processed_rs_eeg_files.txt'  # Track processed files
    overview_file_path = data_dir / 'overview_rs_eeg.csv'  # Path to the overview file

    # Load existing processed files, initialize empty DataFrame if none exist
    try:
        with open(processed_files_path, 'r') as f:
            processed_files = set(f.readlines())
        processed_files = {f.strip() for f in processed_files}
        overview = pd.read_csv(overview_file_path)
    except FileNotFoundError:
        processed_files = set()
        overview = pd.DataFrame(columns=[
            "SubjectId", "FileName", "Condition", "SampleRate", "StartTime",
            "Duration", "NumChannels", "Impedance",
        ])

    # Find all non-DATA .vhdr files
    eeg_rs_files = [f for f in data_dir.glob("**/*.vhdr")]

    for eeg_file in eeg_rs_files:
        if eeg_file.name.strip() not in processed_files:

            print(f"Processing {eeg_file.name}...")
            # Read data with mne
            raw = mne.io.read_raw_brainvision(eeg_file, preload=True)

            # Create new row for overview DataFrame
            new_row = {
                "SubjectId": eeg_file.parent.name,
                "FileName": eeg_file.name,
                "Condition": eeg_file.name.split("_")[-1].split(".")[0],  # Add the condition value here
                "SampleRate": raw.info["sfreq"],  # Use raw.info to access sample rate
                "StartTime": raw.info["meas_date"],  # Use raw.times to access start time
                "Duration": raw.times[-1] - raw.times[0],  # Calculate duration
                "NumChannels ": len(raw.ch_names), # Use len(raw.ch_names) to get number of channels
                "Impedance": ';'.join([str(ch_data["imp"]) for ch_data in raw.impedances.values()]),  # Add the average impedance value here
            }
            overview = pd.concat([overview, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            # Update processed files list
            processed_files.add(eeg_file.name)

            # Optionally save DataFrame to disk (after processing all files)
            overview.to_csv(data_dir / 'overview_rs_eeg.csv', index=False)

        else:
            print(f"Skipping {eeg_file.name} as it has already been processed.")

    # Update processed files text file (outside loop for efficiency)
    with open(processed_files_path, 'w') as f:
        f.writelines('\n'.join(processed_files))

    return overview


def get_info_speed_eeg_files(data_dir):
    """
    Processes walking (.xdf) files in a directory, storing information
    in a growing DataFrame and skipping previously processed files.

    Args:
        data_dir (Path): Path to the directory containing Home Assessment files.

    Returns:
        pandas.DataFrame: The growing DataFrame containing information from processed files.
    """

    processed_files_path = data_dir / 'processed_speed_eeg_files.txt'  # Track processed files
    overview_file_path = data_dir / 'overview_speed_eeg.csv'  # Path to the overview file

    # Load existing processed files, initialize empty DataFrame if none exist
    try:
        with open(processed_files_path, 'r') as f:
            processed_files = set(f.readlines())
        processed_files = {f.strip() for f in processed_files}
        overview = pd.read_csv(overview_file_path)
    except FileNotFoundError:
        processed_files = set()
        overview = pd.DataFrame(columns=[
            "SubjectId", "FileName", "Condition", "SampleRate", "StartTime",
            "Duration", "NumChannels",
        ])

    # Find all non-DATA .cwa files
    eeg_speed_files = [f for f in data_dir.glob("**/*.xdf")]

    for eeg_file in eeg_speed_files:
        if eeg_file.name.strip() not in processed_files:

            print(f"Processing {eeg_file.name}...")
            
            # Read xdf data
            # Identify stream containing EEG channels
            streams = pyxdf.resolve_streams(eeg_file) # raw_fname is an .xdf file
            stream_id = pyxdf.match_streaminfos(streams, [{"type": "EEG"}])
            
            # Get timestamps from datetime created
            timestamp = eeg_file.stat().st_ctime
            start_time = datetime.datetime.fromtimestamp(timestamp)

            # Read in data
            raw = read_raw_xdf(eeg_file, stream_ids = stream_id)
            
            # Create new row for overview DataFrame
            new_row = {
                "SubjectId": eeg_file.parent.name,
                "FileName": eeg_file.name,
                "Condition": eeg_file.name.split("_")[-1].split(".")[0],  # Add the condition value here
                "SampleRate": raw.info["sfreq"],  # Use raw.info to access sample rate
                "StartTime": start_time,  # Use ct_time to access moment file has been created
                "Duration": raw.times[-1] - raw.times[0],  # Calculate duration
                "NumChannels": len(raw.ch_names) - 1, # Use len(raw.ch_names) to get number of channels
            }
            
            overview = pd.concat([overview, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            # Update processed files list
            processed_files.add(eeg_file.name)

            # Optionally save DataFrame to disk (after processing all files)
            overview.to_csv(data_dir / 'overview_speed_eeg.csv', index=False)

        else:
            print(f"Skipping {eeg_file.name} as it has already been processed.")

    # Update processed files text file (outside loop for efficiency)
    with open(processed_files_path, 'w') as f:
        f.writelines('\n'.join(processed_files))

    return overview
