import pandas as pd
import modules.preprocessing.sampling as sampling
import modules.constants as const

import numpy as np
import modules.training.LSTMmodels as LSTMmodels
import torch.nn as nn
import torch.optim as optim
import modules.training.training as training

import modules.utils as utils

from tqdm import tqdm
import os

results_dir = const.casper_results_dir + "Predictions/"
input_dir = const.data_dir + "Casper/Final/"

# Input features
market_df = pd.read_csv(input_dir+ "Market.csv")
accounts_df = pd.read_csv(input_dir + "Accounts.csv")
staking_df = pd.read_csv(input_dir + "Staking.csv")
transactions_df = pd.read_csv(input_dir + "Transactions.csv")
technical_df = pd.read_csv(input_dir + "TI.csv")

y = market_df.current_price.values.reshape(-1,1)

all_dfs = [market_df, accounts_df, staking_df, transactions_df, technical_df]

data_names = ["market", "accounts", "staking", "transactions", "TI"]


## MODEL CONFIGURATION
TEST_SIZE = 0.3
WINDOW_SIZE = 30
STEP_SIZE = 1
OUTPUT_DIM = 1

mse_loss = nn.MSELoss()

N_EPOCHS = 80
N_HIDDEN1 = 256
N_HIDDEN2 = 64
NUM_HEADS = 8
LR = 0.001

BATCH_SIZE = 32

SHORTER_INDEX = 60
DO_SEGMENTATION = True

###################################################
print(f"Running the following configuration: n hidden 1 = {N_HIDDEN1}, n hidden 2 = {N_HIDDEN2}, NUM_HEADS = {NUM_HEADS}, epochs = {N_EPOCHS}, lr = {LR}, batch size = {BATCH_SIZE}, segmentation = {DO_SEGMENTATION}")

# dest_file_results = results_dir + "Results_with_segmentation.csv"
dest_file_predictions = results_dir + "Predictions_complex_with_segmentation.npy"

print("File where predictions are saved: " + dest_file_predictions)

all_results = []
all_predictions = []

# Check if destination files already exist and load their combinations
files_exist = False
done_indexes = []
if os.path.exists(dest_file_predictions):
    files_exist = True
    # prev_metrics = pd.read_csv(dest_file_results)
    prev_predictions = pd.DataFrame(np.load(dest_file_predictions, allow_pickle=True))

    done_indexes = prev_predictions.number.tolist()

for i in tqdm(range(1, 32)):
    # Choose which data combinations to exclude: dont include data set combinations if they do not include either market data or TI data
    if (i < 16 and i % 2 == 0) or (i in done_indexes): continue

    X_data, dict_chosen = utils.get_data_from_combination_number(i, all_dfs, data_names=data_names)
    X_train, y_train, X_test, y_test, scaler = sampling.prepare_input_data(X_data, y, test_size=TEST_SIZE, window_size=WINDOW_SIZE, step_size=STEP_SIZE, do_segmentation=DO_SEGMENTATION)
    train_loader = sampling.make_data_loader(X_train, y_train, batch_size=BATCH_SIZE)

    model = LSTMmodels.LSTMMultiLayerWithAttention(input_dim=X_train.shape[2], hidden_dim1 = N_HIDDEN1, hidden_dim2=N_HIDDEN2, num_heads=NUM_HEADS, output_dim=OUTPUT_DIM)
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
    all_results_df = pd.DataFrame.from_dict(all_results)
    all_predictions_df = pd.DataFrame.from_dict(all_predictions)

    # Convert predictions to numpy format and save numpy file
    if files_exist: 
        # all_results_df = pd.concat([prev_metrics, all_results_df], axis = 0).sort_values(by="number").reset_index(drop=True)
        all_predictions_df = pd.concat([prev_predictions, all_predictions_df], axis = 0).sort_values(by="number").reset_index(drop=True)
    
    # all_results_df.to_csv(dest_file_results, index=False)
    np.save(dest_file_predictions,all_predictions_df.values)

        