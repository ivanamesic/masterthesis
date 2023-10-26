from modules.preprocessing.sampling import CustomLSTMDataset
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


# Train the simple LSTM model using gradient descent. 
# Inputs:
#       - data loader: object of class torch.utils.data.DataLoder which contains input data and target data divided into batches
#       - n epochs: number of epochs that the training will last
#       - model: model which will be trained
#       - loss function: loss function based on which the gradient descent is performed
#       - optimizer: type of optimizer used to perform the training process
#       - print progress: boolean, whether to print intermediate results of loss every 10th epoch
# Outputs:
#       - model: trained model
#       - train loss: array containing loss function results for every epoch
# def train_simple_model(data_loader, n_epochs, model, loss_function, optimizer, print_progress = False):
#     train_loss = []
#     for epoch in range(n_epochs):
#         for batch_data, batch_targets in data_loader:

#             batch_data = batch_data.to(torch.float32)
#             batch_targets = batch_targets.to(torch.float32)

#             optimizer.zero_grad()

#             outputs = model(batch_data)
#             loss = loss_function(outputs, batch_targets)
            
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
        
#         train_loss.append(loss.item())
#         if print_progress and (epoch+1) % 10 == 0:
#                 print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

#     return model, train_loss


# def test_simple_model(X_test, y_test, model, loss_function):
#     predictions = []
#     test_losses = []
#     for i, sample in enumerate(X_test):
#         with torch.no_grad():
#             test_input = torch.Tensor(sample)
#             test_target = torch.Tensor(y_test[i] ) 

#             model_prediction = model(test_input)
#             predictions.append(model_prediction)

#             # Calculate the test loss
#             test_loss = loss_function(model_prediction, test_target)
#             test_loss_scalar = test_loss.item()

#             test_losses.append(test_loss_scalar)

#     return predictions, test_loss_scalar

def train_model(model, data_loader, n_epochs, optimizer, loss_fn, X_val=None, y_val=None, print_progress=False):
    train_loss = []

    do_validation = (X_val is not None) and (y_val is not None)
    val_losses = []

    for epoch in range(n_epochs):
        for batch_data, batch_targets in data_loader:

            batch_data = batch_data.to(torch.float32)
            batch_targets = batch_targets.to(torch.float32)

            optimizer.zero_grad()

            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        if do_validation:
            _, val_loss = make_prediction(model, X_val, y_val, loss_fn=loss_fn)
            val_losses.append(val_loss.item())
        
        train_loss.append(loss.item())
        if print_progress and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

    if do_validation: return model, train_loss, val_losses

    return model, train_loss


def make_prediction(model, X_test, y_test, loss_fn = nn.MSELoss()):
    predictions = []

    for i, sample in enumerate(X_test):
        with torch.no_grad():
            sample_reshaped = sample.reshape(1, X_test.shape[1], X_test.shape[2])

            test_input = torch.Tensor(sample_reshaped)

            model_prediction = model(test_input)
            predictions.append(model_prediction)

            # Calculate the test loss
            # test_loss = loss_fn(model_prediction, test_target)
            # test_loss_scalar = test_loss.item()

            # test_losses.append(test_loss_scalar)
    
    predictions = torch.Tensor(predictions)
    test_targets = torch.Tensor(y_test.flatten()) 
    loss = loss_fn(predictions, test_targets)


    return predictions, loss


def get_all_relevant_metrics(prediction, targets):
    mse = mean_squared_error(y_true = targets, y_pred = prediction)
    rmse = mean_squared_error(y_true = targets, y_pred = prediction, squared= False)
    mae = mean_absolute_error(y_true = targets, y_pred = prediction)
    mape = mean_absolute_percentage_error(y_true = targets, y_pred = prediction)

    result = {
        "MSE": round(mse, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)
    }
    return result




