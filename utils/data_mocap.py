import datetime
import numpy as np
import pandas as pd
import pyxdf

from utils.xdf import get_stream

# find all xdf files in the directory
def get_info_mocap_files(data_dir):
    """
    Processes walking (.xdf) files in a directory, storing information
    in a growing DataFrame and skipping previously processed files.

    Args:
        data_dir (Path): Path to the directory containing Home Assessment files.

    Returns:
        pandas.DataFrame: The growing DataFrame containing information from processed files.
    """

    # Find all .xdf files
    mocap_files = [f for f in data_dir.glob("**/*Speed.xdf")]

    # Initialize empty DataFrame
    overview = pd.DataFrame(columns=[
        "SubjectId", "Visit", "Date", "FileName", "NuniqueMarkerIds", "Duration", "NomSampleRate"
    ])

    for mocap_file in mocap_files:
        # print which file is being processed
        print(f"Processing {mocap_file.name}...")

        # Read xdf data
        data, header = pyxdf.load_xdf(mocap_file)

        mocap = get_stream(data, "Qualisys")

        # Get metadata
        subject_id = mocap_file.name.split("_")[1:3]
        subject_id = str(subject_id[0] + "_" + subject_id[1])
        visit = mocap_file.name.split("_")[3]
        timestamp = mocap_file.stat().st_ctime
        date = datetime.datetime.fromtimestamp(timestamp)

        # Get duration
        duration = mocap["time_stamps"][-1] - mocap["time_stamps"][0]

        # Get sample rate
        sample_rate = mocap["info"]["nominal_srate"][0]

        # Get the number of unique marker ids
        marker_ids =  mocap["time_series"][:, -1]
        unique_marker_ids = np.unique(marker_ids)

        # Create new row for overview DataFrame
        new_row = {
            "SubjectId": subject_id,
            "Visit": visit,
            "Date": date,
            "FileName": mocap_file.name,
            "NuniqueMarkerIds": len(unique_marker_ids),
            "Duration": duration,
            "NomSampleRate": sample_rate,
        }
        overview = pd.concat([overview, pd.DataFrame(new_row, index=[0])], ignore_index=True)

    return overview