# Reproducibility Guide: Quarterly Retraining and Performance Evaluation of Financial Time Series Models

This repository contains the code necessary to reproduce the methodology presented in the thesis, covering data preparation, risk-neutral path generation, and final performance assessment.

## 1. Nelson-Siegel Model Fitting

This step involves fitting the Nelson-Siegel model to market data (e.g., bond yields) to extract the key parameters (Level, Slope, Curvature) that serve as the primary input features for the subsequent GAN models.

* **File name:** `GIT_NELSON_SIEGEL.py`
  
## 2. Data Filtering (Preprocessing)

Data filtering stabilizes the data distribution and removes outliers, ensuring robust performance during the GAN training phase.

* **File name:** `GIT_DATA_FILTER.py`
* **Key Functionality:**
    * Data used for pricing.
    * All filters used in the paper were applied.

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

**NOTE ON IMPLEMENTATION:** The complex, hundreds-of-lines implementation code for the TimeGAN, QuantGAN, and SigCWGAN models themselves is **omitted** from this repository. These models are based on established, externally available research libraries cited in the paper.



## 5. Risk-Neutral Path Generation and Price/Delta Estimation

Using the generated noised by each GAN models, this stage generates risk-neutral paths and estimates option prices and Delta values.

* **File name:** `GIT_PRICING.py`
* **Key Functionality:**
    * Loads the 39 pkl files from `noise_path/` .
    * 

## 6. Model Result Aggregation

All estimated prices and deltas from the 39 quarters and multiple models are collected into a single, comprehensive dataset.

* **Code Location:** `src/analysis/aggregate_results.py`
* **Key Functionality:**
    * Loads simulation results from `results/simulations/`.
    * Merges all model results and the actual 'real' target values into a single `res_df` DataFrame, preparing the data for the final statistical evaluation.

## 7. Performance Evaluation (Multiple Comparison with the Best - MCS)

This is the final step where the statistical superiority of the proposed models against the benchmarks is rigorously verified, aligning with the empirical claims of the paper.

* **Code Location:** `src/analysis/performance_metrics.py` (Main function: `yearly_mcs`)
* **Key Functionality:**
    * **Loss Calculation:** Calculates the sequential loss metrics (e.g., Squared Error) for each model across the entire testing period.
    * **MCS Test:** Applies the MCS procedure (using the `arch` library) to the **time series of losses** (not averaged losses), providing the necessary statistical rigor to select the best-performing set of models at a given confidence level.


