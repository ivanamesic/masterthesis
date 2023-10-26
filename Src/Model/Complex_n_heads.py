# %% [markdown]
# # HYPERPARAMETER TUNING - COMPLEX MODEL

# %%
import pandas as pd
import modules.preprocessing.sampling as sampling
import modules.preprocessing.scaling as scaling
import modules.constants as const

import numpy as np
import modules.training.LSTMmodels as LSTMmodels
import torch.nn as nn
import torch.optim as optim
import modules.training.training as training

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import torch

import modules.utils as utils
import modules.plot_utils as plutils
import modules.plot_constants as pltconst
from modules.plot_constants import uzh_colors

from tqdm import tqdm
import os

# Load input data sets
market_df = pd.read_csv(const.input_X_dir + "Market.csv")
network_df = pd.read_csv(const.input_X_dir + "NetworkActivity.csv")
social_df = pd.read_csv(const.input_X_dir + "SocialNetworks.csv")
supply_df = pd.read_csv(const.input_X_dir + "Supply.csv")
technical_df = pd.read_csv(const.input_X_dir + "TechnicalIndicators.csv")

# Target feature and dates
df_y = pd.read_csv(const.input_y_dir + "Target.csv")
dates_df = pd.read_csv(const.input_y_dir + "Dates.csv")

y = df_y.values
dates = dates_df.values.flatten()

figures_dir = const.tezos_results_dir + "Hyperparameter tuning/Figures/"
tables_dir = const.tezos_results_dir + "Hyperparameter tuning/Tables/"


N_VALIDATION_SPLITS = 4

LR = 0.001

N_HIDDEN1 = 256
N_HIDDEN2 = 64
N_EPOCHS = 80
mse_loss = nn.MSELoss()

TEST_SIZE = sampling.calculate_test_size_from_date(const.test_start_date)
WINDOW_SIZE = 30
STEP_SIZE = 1
OUTPUT_DIM = 1
BATCH_SIZE = 128

attention_head_options= [4, 8, 16]

X = market_df.values

n_features = X.shape[1]
X_train, y_train, X_test, y_test, scaler = sampling.prepare_input_data(X, y, test_size=TEST_SIZE, window_size=WINDOW_SIZE, step_size=STEP_SIZE, do_segmentation=False)

tscv = TimeSeriesSplit(n_splits=N_VALIDATION_SPLITS)
dest_file = tables_dir + "Complex/Complex_num_heads.csv"

# Iterate through the splits and perform training/testing
results = []

file_exists = os.path.exists(dest_file)
if file_exists: 
    prev_results = pd.read_csv(dest_file)

print(f"Running the following configuration: n hidden 1 = {N_HIDDEN1}, n hidden 2 = {N_HIDDEN2}, epochs = {N_EPOCHS}, lr = {LR}, batch size = {BATCH_SIZE}")
for num_heads in attention_head_options:
        # if file_exists:
        #     tgt = prev_results[(prev_results.num_heads == num_heads)]
        #     if tgt.shape[0] > 0: continue

        result_row = { "num_heads": num_heads }
        tr_loss, val_loss = [], []
        training_curves, validation_curves = [], []
        
        # Iterate over blocked validation splits
        for train_indexes, val_indexes in tqdm(tscv.split(X_train)):
            X_tr, y_tr, X_val, y_val = X_train[train_indexes], y_train[train_indexes], X_train[val_indexes], y_train[val_indexes]

            model = LSTMmodels.LSTMMultiLayerWithAttention(input_dim=X_tr.shape[2], hidden_dim1 = N_HIDDEN1, hidden_dim2=N_HIDDEN2, num_heads=num_heads, output_dim=OUTPUT_DIM)
            optimizer = optim.Adam(model.parameters(), lr=LR)

            data_loader = sampling.make_data_loader(X_tr, y_tr, batch_size=BATCH_SIZE)

            model, train_loss_curve, validation_loss_curve = training.train_model(model, data_loader, n_epochs=N_EPOCHS, optimizer=optimizer, loss_fn = mse_loss, X_val=X_val, y_val=y_val)

            training_curves.append(train_loss_curve)
            validation_curves.append(validation_loss_curve)
            predictions, val_loss_value = training.make_prediction(model, X_val, y_val, mse_loss)

            tr_loss.append(train_loss_curve[-1])
            val_loss.append(val_loss_value)

        result_row["train_loss"]= np.average(tr_loss)
        result_row["validation_loss"] = np.average(val_loss)

        result_row["training_curve"] = np.average(np.array(training_curves), axis = 0)
        result_row["validation_curve"] = np.average(np.array(validation_curves), axis = 0)

        results.append(result_row)

        if len(results) == 0: continue
        df1 = pd.DataFrame.from_dict(results).sort_values(by="validation_loss", ascending=True).reset_index(drop=True)
        if file_exists:
            df1 = pd.concat([prev_results, df1], axis= 0)
        df1.to_csv(dest_file, index=False)



