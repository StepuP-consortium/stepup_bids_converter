# StepuP2BIDS: EMG, EEG and MoCap Data Conversion to BIDS Format

## Description
This repository contains scripts and data for converting electromyography (EMG), electroencephalography (EEG), and motion data from a custom MATLAB format to Brain Imaging Data Structure (BIDS) format using the FieldTrip toolbox. The data conversion facilitates the analysis of neurological and movement data in a standardized format, suitable for further analysis and sharing within the scientific community.

## Data Description
The dataset includes the following components:
- **EMG Data**: Electromyographic data from 14 channels.
- **EMG Labels**: Names of the EMG channels.
- **L_heel and R_heel**: Data from sensors placed on the left and right heels, respectively.
- **Pelvis**: Data from a sensor placed on the pelvis.
- **data_EEG**: Structured EEG data.
- **events_EEG, events_EMG, events_qls**: Event structures corresponding to EEG, EMG, and qualitative life signs (qls).
- **fs_qls**: Sampling rate for the qls data.
- **t_qls**: Time vector for qls data.

## Repository Structure
```
/
|-- data/
|   |-- source/        # Original MATLAB data files
|   |-- bids/            # Converted BIDS format data
|
|-- scripts/
|   |-- pilot_converter_amsterdam.m  # MATLAB script to convert data to BIDS format
|
|-- README.md            # This readme file
```

## Prerequisites
To run the scripts provided in this repository, you need:
- MATLAB (recommended version R2021a or later)
- FieldTrip toolbox, installed and configured in your MATLAB environment

## Usage
1. Clone this repository to your local machine.
2. Ensure your MATLAB path includes the FieldTrip toolbox.
3. Place your original MATLAB `.mat` files in the `data/source/` directory.
4. Run the script `convert_to_bids.m` from the MATLAB command window. This script reads the data from the `data/source/` directory, processes it, and writes the BIDS-compatible output to the `data/bids/` directory.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit pull requests with any enhancements, bug fixes, or improvements.

## License
Specify the license under which this repository is made available, such as MIT, GPL, etc.

## Contact
For any queries regarding this project, please contact [Julius Welzel](mailto:julius.welzel@gmail.com?subject=StepuP%20BIDS%20converter).
