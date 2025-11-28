# Reproducibility Guide: Quarterly Retraining and Performance Evaluation of Financial Time Series Models

This repository contains the code necessary to reproduce the methodology presented in the thesis, covering data preparation, risk-neutral path generation, and final performance assessment.

## 1. Nelson-Siegel Model Fitting

This step involves fitting the Nelson-Siegel model to market data (e.g., bond yields) to extract the key parameters (Level, Slope, Curvature) that serve as the primary input features for the subsequent GAN models.

* **File name:** `GIT_NELSON_SIEGEL.py`
* * **Key Functionality:**
    * Raw price data is located at 'data/raw/sp500.csv'.
    * Final risk-free rate dataset is saved at 'data/processed/rfcurve.csv' (not included in this repository due to file size).
  
## 2. Data Filtering (Preprocessing)

Data filtering stabilizes the data distribution and removes outliers, ensuring robust performance during the GAN training phase.

* **File name:** `GIT_DATA_FILTER.py`
* **Key Functionality:**
    * Data used for pricing.
    * All filters used in the paper were applied.
    * Raw sample dataset is located at 'data/raw/opdat_raw.pkl'
    * Filtered dataset is located at 'data/processed/opdat.pkl'

## 3. Training Set Construction

This is a critical step for implementing the quarterly retraining methodology. The complete time series is segmented into 39 sequential training datasets.

* **File name:** `GIT_TRAIN_SET.py`
* **Key Functionality:**
    * Divides the full time series data into 39 discrete quarterly intervals.
    * Train dataset location : 'data/trainset'
    * Each segmented dataset is saved as a separate CSV file, named sequentially (e.g., `data/trainset/train0.csv` through `data/trainset/train38.csv`).
    * **Required for Reproduction:** The 39 CSV files must be placed exactly within the `data/trainset/` folder.

## 4. GAN Retraining Loop (Implementation Omitted)

This section demonstrates the core methodology: sequentially retraining TimeGAN, QuantGAN, and SigCWGAN on the 39 quarterly datasets.
**Key Functionality:**
    * **Required for Reproduction:** The 39 pkl files for 10000 paths of 91 step noises that genereated by each GAN models must be saved.
    * **Required for Reproduction:** This repository do not include real generated noises.

**NOTE ON IMPLEMENTATION:** The complex, hundreds-of-lines implementation code for the TimeGAN, QuantGAN, and SigCWGAN models themselves is **omitted** from this repository. These models are based on established, externally available research libraries cited in the paper.



## 5. Risk-Neutral Path Generation and Price/Delta Estimation

Using the generated noised by each GAN models, this stage generates risk-neutral paths and estimates option prices and Delta values.

* **File name:** `GIT_PRICING.py`
* **Key Functionality:**
    * Loads the 39 pkl files from `noise_path/` .
    * For each option contract, pricing is performed using the period-matching noise file from the 39 datasets (which were pre-processed using option data columns).
    * The pricing resutls of each models are saved at 'results/Price/modelname*/optiontype*' or 'results/Delta/modelname*/optiontype*'

## 6. Model Result Aggregation

All estimated prices and deltas from the 39 quarters and multiple models are collected into a single, comprehensive dataset.

* **File nmae:** `GIT_RESULTS.py`
* **Key Functionality:**
    * Aggregate all models' results
    * The HESTON model code and results are currently being organized and will be updated soon.

## 7. Performance Evaluation (Multiple Comparison with the Best - MCS)

This is the final step where the statistical superiority of the proposed models against the benchmarks is rigorously verified, aligning with the empirical claims of the paper.

* **File name:** `GIT_PERFORMANCE.py`)
* **Key Functionality:**
    * Overall & Sub-period performance calculation code is included.

**The code supporting the full thesis is undergoing continuous refinement and will be subject to ongoing updates.**

