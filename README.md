# Reconstruction-for-Remote-Sensing-Time-Series
Remote sensing time-series missing data reconstruction algorithm implementation.

## Project Overview
This repository contains source codes of the submitted papper "Trend-aware Reconstruction for Remote Sensing Time Series under Non-random
Missingness"

## Data Preparation
1. Download MOD13QI and put the dataset into the `MOD13QI/` folder under the project root path.
2. Run `build_data.py` to convert the original dataset into formatted time series data:
```bash
python build_data.py
```
3. Run `split_dataset.py` to split the processed data into training, validation, and test sets:
```bash
python split_dataset.py
```

## Training
Start model training using the processed dataset:
```bash
python train.py
```
**The model implementation code will be publicly available after the paper is accepted.**

## Testing
Run the test script to evaluate model performance on test dataset:
```bash
python test.py
```
