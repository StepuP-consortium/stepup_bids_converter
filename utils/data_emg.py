import datetime
from mnelab.io.xdf import read_raw_xdf
import pandas as pd
import pyxdf

from utils.xdf import get_effective_srate_xdf


def get_info_emg_files(data_dir):
    """
    Processes walking (.xdf) files in a directory, storing information
    in a growing DataFrame and skipping previously processed files.

    Args:
        data_dir (Path): Path to the directory containing Home Assessment files.

    Returns:
        pandas.DataFrame: The growing DataFrame containing information from processed files.
    """

    processed_files_path = data_dir / 'processed_emg_files.txt'  # Track processed files
    overview_file_path = data_dir / 'overview_emg.csv'  # Path to the overview file

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
            "Duration", "NumChannels","NumSamples"
        ])

    # Find all non restingstate .xdf files
    emg_files = [f for f in data_dir.glob("**/*.xdf")]
    emg_files = [f for f in emg_files if "restingstate" not in f.name]

    for emg_file in emg_files:
        if emg_file.name.strip() not in processed_files:

            print(f"Processing {emg_file.name}...")
            
            # Read xdf data
            # Identify stream containing EEG channels
            lsl_streams = pyxdf.resolve_streams(emg_file) # raw_fname is an .xdf file
            emg_stream_id = pyxdf.match_streaminfos(lsl_streams, [{"type": "EMG"}])
            
            # Get timestamps from datetime created
            timestamp = emg_file.stat().st_ctime
            start_time = datetime.datetime.fromtimestamp(timestamp)

            # load raw emg
            emg_raw, _ = pyxdf.load_xdf(emg_file, select_streams=emg_stream_id, dejitter_timestamps=False)


            effective_srate = get_effective_srate_xdf(emg_raw)

            # Create new row for overview DataFrame
            new_row = {
                "SubjectId": emg_file.parent.name,
                "FileName": emg_file.name,
                "Condition": emg_file.name.split("_")[-1].split(".")[0],  # Add the condition value here
                "SampleRate": float(emg_raw[0]["info"]["nominal_srate"][0]),  # Use raw.info to access sample rate
                "SampleRateEffective": effective_srate,  # Use raw.info to access sample rate
                "StartTime": start_time,  # Use ct_time to access moment file has been created
                "Duration": emg_raw[0]["time_stamps"][-1] - emg_raw[0]["time_stamps"][0],  # Calculate duration
                "NumChannels": int(emg_raw[0]["info"]["channel_count"][0]), # Use len(raw.ch_names) to get number of channels
                "NumSamples": np.shape(emg_raw[0]["time_series"])[0] # Use len(raw.times) to get number of samples
            }
            
            overview = pd.concat([overview, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            # Update processed files list
            processed_files.add(emg_file.name)

            # Optionally save DataFrame to disk (after processing all files)
            overview.to_csv(data_dir / 'overview_emg.csv', index=False)

        else:
            print(f"Skipping {emg_file.name} as it has already been processed.")

    # Update processed files text file (outside loop for efficiency)
    with open(processed_files_path, 'w') as f:
        f.writelines('\n'.join(processed_files))

    return overview

