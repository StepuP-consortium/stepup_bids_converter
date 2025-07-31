import pandas as pd

# generate emg.json file for the emg files in the BIDS dataset
def generate_emg_json_file(required_fields: dict, optional_fields: dict = {}):
    """
    Generate a emg.json file based on the provided schema.
    
    Parameters:
    required_fields (dict): A dictionary containing the required fields and their values.
                            The dictionary must include the following keys:
                            - "TaskName" (str): Name of the task.
                            - "SamplingFrequency" (float): Sampling frequency of the emg data.
    optional_fields (dict): A dictionary containing the optional fields and their values.
                            This can include any additional metadata fields as key-value pairs.
    
    Returns:
    dict: A dictionary containing the combined required and optional fields.
    
    Raises:
    ValueError: If any required field is missing.
    """
    required_keys = ["EMGPlacementScheme", "EMGReference", "SamplingFrequency", "PowerLineFrequency", "SoftwareFilters", "TaskName"]
    
    # Check if all required fields are provided
    for key in required_keys:
        if key not in required_fields:
            raise ValueError(f"Missing required field: {key}")
    
    # Combine required and optional fields
    emg_data = {**required_fields, **optional_fields}

    # Return the emg data as a dict
    return emg_data

# generate a channels.tsv file for the emg files in the BIDS dataset
def generate_channels_tsv(channels_name: list, sampling_rate: int = 100):
    """
    Generate a TSV file for channels based on tracked markers.
    Parameters:
    tracked_markers (list): A list of strings representing the tracked markers. 
                            Each marker should be in lower case and contain no underscores.
    sampling_rate (int): The sampling rate for the channels. Default is 100.
    Returns:
    pandas.DataFrame: A DataFrame containing the channel information with the following columns:
                      - name: Channel names (marker_x, marker_y, marker_z for each marker).
                      - components: Channel components (x, y, z for each marker).
                      - type: Channel types (POS for each marker).
                      - tracked_points: Tracked points labels (each marker name repeated 3 times).
                      - units: Channel units (m for each marker).
                      - sampling_rate: Channel sampling rates (sampling_rate for each marker).
    Raises:
    ValueError: If tracked_markers is not a list of strings or if any marker is not lower case or contains underscores.
    """
    if not isinstance(channels_name, list) or not all(isinstance(marker, str) for marker in channels_name):
        raise ValueError("tracked_markers should be a list of strings")
    if not all("_" not in marker for marker in channels_name):
        raise ValueError("All markers should contain no underscores")

    # create channel names in name column (tracked markers * 3)
    channel_names = []
    for marker in channels_name:
        channel_names.append(f"{marker}_x")
        channel_names.append(f"{marker}_y")
        channel_names.append(f"{marker}_z")
        
    # create channel components 
    channel_components = []
    for _ in channels_name:
        channel_components.append("x")
        channel_components.append("y")
        channel_components.append("z")

    # create channel types
    channel_types = []
    for _ in channels_name:
        channel_types.append("POS")
        channel_types.append("POS")
        channel_types.append("POS")
        
    # create channel tracked_points labels, each name should appear 3 times for x,y,z
    tracked_points = []
    for marker in tracked_markers:
        tracked_points.extend([marker] * 3)
        
    # create channel units
    channel_units = []
    for _ in tracked_markers:
        channel_units.append("mm")
        channel_units.append("mm")
        channel_units.append("mm")

    # return as a pandas dataframe
    channels_tsv = pd.DataFrame({
        "name": channel_names,
        "component": channel_components,
        "type": channel_types,
        "tracked_point": tracked_points,
        "units": channel_units,
    })
    
    return channels_tsv