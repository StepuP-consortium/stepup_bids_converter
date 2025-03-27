import actipy
import pandas as pd

def get_info_ha_files(data_dir):
    """
    Processes Home Assessment (.cwa) files in a directory, storing information
    in a growing DataFrame and skipping previously processed files.

    Args:
        data_dir (Path): Path to the directory containing Home Assessment files.

    Returns:
        pandas.DataFrame: The growing DataFrame containing information from processed files.
    """

    processed_files_path = data_dir / 'processed_ha_files.txt'  # Track processed files

    # Load existing processed files, initialize empty DataFrame if none exist
    try:
        with open(processed_files_path, 'r') as f:
            processed_files = set(f.readlines())
    except FileNotFoundError:
        processed_files = set()
        overview = pd.DataFrame(columns=[
            "SubjectId", "FileName", "DeviceID", "SampleRate", "StartTime",
            "EndTime", "Duration", "DurationDeviceOff", "NumInterrupts", "ReadErrors"
        ])

    # Find all non-DATA .cwa files
    ha_files = [f for f in data_dir.glob("**/*.cwa") if "DATA" not in f.name]

    for ha_file in ha_files:
        if ha_file.name not in processed_files:
            # Read data, extract information from Axivity data file
            data, info = actipy.read_device(str(ha_file), lowpass_hz=20, calibrate_gravity=True)

            # Create new row for overview DataFrame
            new_row = {
                "SubjectId": ha_file.parent.name,
                "FileName": ha_file.name,
                "DeviceID": info["DeviceID"],
                "SampleRate": info["SampleRate"],
                "StartTime": info["StartTime"],
                "EndTime": info["EndTime"],
                "Duration": info["WearTime(days)"],
                "DurationDeviceOff": info["NonwearTime(days)"],
                "NumInterrupts": info["NumInterrupts"],
                "ReadErrors": info["ReadErrors"]
            }
            overview = pd.concat([overview, pd.DataFrame(new_row, index=[0])], ignore_index=True)

            # Update processed files list
            processed_files.add(ha_file.name)

            # Optionally save DataFrame to disk (after processing all files)
            overview.to_csv(data_dir / 'overview_ha.csv', index=False)

        else:
            print(f"Skipping {ha_file.name} as it has already been processed.")

    # Update processed files text file (outside loop for efficiency)
    with open(processed_files_path, 'w') as f:
        f.writelines('\n'.join(processed_files))

    return overview