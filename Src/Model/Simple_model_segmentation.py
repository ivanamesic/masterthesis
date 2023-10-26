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

results_dir = const.tezos_results_dir + "Predictions/Simple Model/"

# Input features
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

all_dfs = [market_df, network_df, social_df, supply_df, technical_df]

data_names = ["market", "network", "social", "supply", "TI"]

TEST_SIZE = sampling.calculate_test_size_from_date(const.test_start_date)
WINDOW_SIZE = 30
STEP_SIZE = 1
OUTPUT_DIM = 1

mse_loss = nn.MSELoss()

N_EPOCHS = 80
N_HIDDEN = 256
LR = 0.001

BATCH_SIZE = 32
DO_SEGMENTATION = True

SHORTER_INDEX = 60

dest_file_results = results_dir + "Results_with_segmentation256.csv"
dest_file_predictions = results_dir + "Predictions_with_segmentation256.npy"

all_results = []
all_predictions = []
print(f"Running the following configuration: n hidden 1 = {N_HIDDEN}, epochs = {N_EPOCHS}, lr = {LR}, batch size = {BATCH_SIZE}, segmentation = {DO_SEGMENTATION}")

for i in tqdm(range(1, 32)):
    # Choose which data combinations to exclude: dont include data set combinations if they do not include either market data or TI data
    if i < 16 and i % 2 == 0: continue

    X_data, dict_chosen = utils.get_data_from_combination_number(i, all_dfs, data_names=data_names)
    X_train, y_train, X_test, y_test, scaler = sampling.prepare_input_data(X_data, y, test_size=TEST_SIZE, window_size=WINDOW_SIZE, step_size=STEP_SIZE, do_segmentation=DO_SEGMENTATION)
    train_loader = sampling.make_data_loader(X_train, y_train, batch_size=BATCH_SIZE)

    model = LSTMmodels.LSTMSimple(input_size=X_train.shape[2], hidden_size=N_HIDDEN, output_size=OUTPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Train the model
    model, _ = training.train_model(model, train_loader, n_epochs=N_EPOCHS, optimizer=optimizer, loss_fn = mse_loss)
    
    # Predict on the full test set and unscale the values
    predictions, _ = training.make_prediction(model, X_test, y_test, mse_loss)
    predictions_unsc = scaler.inverse_transform(predictions.reshape(-1,1))

    # Meaasure metrics for both test sets
    metrics1 = training.get_all_relevant_metrics(prediction=predictions_unsc, targets=y_test)
    metrics2 = training.get_all_relevant_metrics(prediction=predictions_unsc[:SHORTER_INDEX], targets=y_test[:SHORTER_INDEX])

    # Append the results of this data combination
    metrics2 = {key + "_short": value for key, value in metrics2.items()}
    
    all_results.append({**dict_chosen, **metrics1, **metrics2})
    all_predictions.append({**dict_chosen, **{"prediction": predictions_unsc.flatten()}})

    # Save the metrics results tables
    pd.DataFrame.from_dict(all_results).to_csv(dest_file_results, index=False)

    # Convert predictions to numpy format and save numpy file
    all_predictions_df = pd.DataFrame.from_dict(all_predictions)
    np.save(dest_file_predictions,all_predictions_df.values)

        