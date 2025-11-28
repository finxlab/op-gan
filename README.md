eproducibility Guide: Quarterly Retraining and Performance Evaluation of Financial Time Series Models

This repository contains the code necessary to reproduce the methodology presented in the thesis, covering data preparation, quarterly model retraining, risk-neutral path generation, and final performance assessment.

1. Nelson-Siegel Model Fitting

This step involves fitting the Nelson-Siegel model to market data (e.g., bond yields) to extract the key parameters (Level, Slope, Curvature) that serve as the primary input features for the subsequent GAN models.

Code Location: src/data_processing/nelson_siegel_fitting.py

Key Functionality:

Loads and cleans raw market data.

Extracts Nelson-Siegel parameters to generate the input time series.

Prepares the essential time series used as features for the GAN models.

2. Data Filtering (Preprocessing)

Data filtering stabilizes the data distribution and removes outliers, ensuring robust performance during the GAN training phase.

Code Location: src/data_processing/filtering_and_scaling.py

Key Functionality:

Performs outlier detection and treatment (e.g., using IQR or 3-Sigma Rule).

Applies differencing, if necessary, to ensure time series stationarity.

Scales the final preprocessed data (e.g., MinMax or Standard Scaling) ready for model consumption.

3. Training Set Construction

This is a critical step for implementing the quarterly retraining methodology. The complete time series is segmented into 39 sequential training datasets.

Code Location: src/data_processing/build_trainsets.py

Key Functionality:

Divides the full time series data into 39 discrete quarterly intervals.

Each segmented dataset is saved as a separate CSV file, named sequentially (e.g., data/trainsets/quarter_1.csv through quarter_39.csv).

Required for Reproduction: The 39 CSV files must be placed exactly within the data/trainsets/ folder.

4. GAN Retraining Loop (Implementation Omitted)

This section demonstrates the core methodology: sequentially retraining TimeGAN, QuantGAN, and SigCWGAN on the 39 quarterly datasets.

NOTE ON IMPLEMENTATION: The complex, hundreds-of-lines implementation code for the TimeGAN, QuantGAN, and SigCWGAN models themselves is omitted from this repository. These models are based on established, externally available research libraries cited in the paper.

Provided Code (Orchestration):

src/timegan_retraining.py

src/quantgan_retraining.py

src/sigcwgan_retraining.py

Demonstrated Logic: These scripts contain the training loop that iterates from Quarter 1 to Quarter 39, ensuring the correct data is loaded, the hyperparameters specified in the paper are used, and the trained weights are saved to results/models/ for subsequent simulation. This loop validates the methodology.

5. Risk-Neutral Path Generation and Price/Delta Estimation

Using the quarterly trained models, this stage generates risk-neutral paths for Monte Carlo simulation and estimates derivative prices and Delta values.

Code Location: src/simulation/rn_path_generation.py

Key Functionality:

Loads the 39 trained GAN models from results/models/.

Utilizes the Generator component of the models to create synthetic time series paths under a risk-neutral measure.

Performs Monte Carlo estimation on the generated paths to calculate final option prices and corresponding Delta values.

6. Model Result Aggregation

All estimated prices and deltas from the 39 quarters and multiple models are collected into a single, comprehensive dataset.

Code Location: src/analysis/aggregate_results.py

Key Functionality:

Loads simulation results from results/simulations/.

Merges all model results and the actual 'real' target values into a single res_df DataFrame, preparing the data for the final statistical evaluation.

7. Performance Evaluation (Multiple Comparison with the Best - MCS)

This is the final step where the statistical superiority of the proposed models against the benchmarks is rigorously verified, aligning with the empirical claims of the paper.

Code Location: src/analysis/performance_metrics.py (Main function: yearly_mcs)

Key Functionality:

Loss Calculation: Calculates the sequential loss metrics (e.g., Squared Error) for each model across the entire testing period.

MCS Test: Applies the MCS procedure (using the arch library) to the time series of losses (not averaged losses), providing the necessary statistical rigor to select the best-performing set of models at a given confidence level.

Steps to Reproduce the Analysis

git clone [Repository URL]

pip install -r requirements.txt (Install dependencies)

Place the 39 training datasets (quarter_1.csv, etc.) into the data/trainsets/ folder.

Execute the scripts in the following order:

python src/timegan_retraining.py (GAN 1 Training)

python src/quantgan_retraining.py (GAN 2 Training)

python src/sigcwgan_retraining.py (GAN 3 Training)

python src/simulation/rn_path_generation.py (Simulation and Estimation)

python src/analysis/performance_metrics.py (Final MCS Evaluation)
