# StepuP2BIDS: EMG, EEG and MoCap Data Conversion to BIDS Format

## Description
This repository contains scripts and data for converting electromyography (EMG), electroencephalography (EEG), and motion data from a custom MATLAB format to Brain Imaging Data Structure (BIDS) format using the FieldTrip toolbox. 
The code has been updated in November 2024 to cover the content of the workshop held in Zurich.


## Data Description
To download the data please visit the [link](https://cloud.rz.uni-kiel.de/index.php/s/rjTPpC2dKwFjiW8) to the repository from Kiel University.
There are two datasets available:
- **S1_CS**: This dataset contains EMG, EEG, and motion data from a healthy participants recorded in Amsterdam. 
- **data_DEU_F57fyw_T1_FixSpeed**: This dataset contains EMG, EEG, and motion data from a parkinson patients recorded in Kiel.

The dataset includes the following components:
    - **EEG Data**: Electromyographic data from 128 or 64 channels.
    - **EMG Data**: Electromyographic data from 12 or 6 channels.
    - **Motion Data**: Motion data from Qualysis of Vicon systems of pelvis, left and right foot.

## Repository Structure
Please make sure to have the following structure in your repository and place the source data files in the `source` data directory.
```
/
|-- data/
|   |-- source/        # Original  data files
|   |-- bids/            # Converted BIDS format data
|
|-- scripts/
|   |-- pilot_converter_amsterdam.m  # MATLAB script to convert data to BIDS format
|   |-- pilot_converter_kiel.m       # MATLAB script to convert data to BIDS format
|
|-- utils_matlab/
|   |-- prep_omc_bids.m  # Helper function to prepare motion capture data for BIDS conversion
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
3. Place your original MATLAB `.mat` or XDF `.xdf` files in the `data/source/` directory.
4. Run any script `*converter*.m` from the MATLAB command window. This script reads the data from the `data/source/` directory, processes it, and writes the BIDS-compatible output to the `data/bids/` directory.

## Contact
For any queries regarding this project, please contact [Julius Welzel](mailto:julius.welzel@gmail.com?subject=StepuP%20BIDS%20converter).
