# STEAM_RUL_Prediction
The Python files in this project implement two battery life prediction models: one uses charging current as the key variable, and the other uses temperature as the key variable. Each model is trained on a different dataset and saved separately.


1. The implementation-oriented description of the code contents and their execution order for the current-variable (charging current) model is as follows:
(The dataset used to train the model was the open-source dataset released alongside the paper: Severson et al. Data-driven prediction of battery cycle life before capacity degradation, nature energy, 2019)

1-1. data_split.py
Segregates all CSV files located under the 'data\all' directory into training and test datasets and saves them accordingly.

2-1. train_lgbm.py
Trains a LightGBM model that takes charging-protocol information as input and predicts four features(voltage, internal resistance, the mean of dQ/dV, and the variance of dQ/dV).
The trained model is saved to 'models\lgbm'.

3-1. train.py
Trains a 1D CNN-LSTM model. The trained model is exported to the 'export' directory.

4-1. test.py
Uses the saved model to select three datasets from the test set defined within the code, generates predictions, and saves the resulting plots.


2. The implementation-oriented description of the code contents and their execution order for the temperature-variable model is as follows:
(The dataset used to train the model was the open-source dataset released by NASA: B. Saha and K. Goebel (2007). “Battery Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA)

This Python code targets battery life prediction with temperature as a key variable: in particular, temp.py defines a general empirical model for all CSV files in the 'data' directory and fits its coefficients using Hamiltonian Monte Carlo (HMC) to estimate their posterior distributions.
The model is Qd = 2 - (a*T*N)*exp(d*I) - exp(b*(V-1))*exp(c*N), where Qd denotes discharge capacity, T ambient temperature, N the charge/discharge cycle count, I discharge current, and V discharge voltage, with a,b,c,d being the coefficients to be inferred. Prediction results generated from this model are saved to the 'plot' directory.
