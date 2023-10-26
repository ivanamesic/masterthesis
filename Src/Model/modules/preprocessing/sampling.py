import modules.constants as const
import modules.preprocessing.scaling as scaling
import numpy as np

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader


# This class is a custom dataset class made for easier loading of the data for LSTM.
class CustomLSTMDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Split the original data into a train and test subset. The input data should be a numpy array. 
# If a pandas dataframe is given, it has to be specified with setting is_df to True. The result is always a numpy array.
# Inputs:
#       - data: data that needs to be split. the split is performed chronologically, so the first n elements are the train set and the rest is the test set. 
#               Data can be either pandas df (is_df has to be set to True) or a numpy array (default setting).
#       - test_size: percentage of the entire dataset which will be used as the test set. given in decimal value
#       - is_df: boolean. If true, the input data is considered to be a pandas dataframe
# Output:
#       - train, test: the train set and the test set
def get_train_test_split(data, test_size, is_df = False):
    if not isinstance(data, np.ndarray):
        raise ValueError("The input data should be type numpy array")
    
    index = int(data.shape[0]*(1-test_size))
    
    return data[:index], data[index:]

# Calculate the size of the test set (in percentage of the whole data set) from the start date where the test data set starts
def calculate_test_size_from_date(test_start_date):
    date_ind = list(const.date_range).index(test_start_date)
    test_size = 1 - (date_ind / len(const.date_range))
    return round(test_size, 2)


# Apply the sliding window technique. This technique slides the window of size "window_size" in increments of "step_size" through the input data. 
# The target variable corresponding to the input window is the first target value after the last element in the window.
# Inputs:
#       - X: input data
#       - y: target variable
#       - window size: size of the sliding window; number of time points included in a sample
#       - step size: number of time points between consecutive windows; time step of the "slide" movement
# Outputs:
#       - input sequences: list of "windows" of data collected
#       - targets: list of target variables corresponding to each window of data
def apply_sliding_window(X, y, window_size=7, step_size=1):
    if X.shape[0] != len(y):
        raise ValueError("Input and target variables have to be of same length!")

    input_sequences = []
    targets = []

    # Create data using a sliding window
    for i in range(0, len(X) - window_size , step_size):
        window_data = X[i:i + window_size, ]
        target = y[i + window_size]
        
        input_sequences.append(window_data)
        targets.append(target)

    # Convert the lists to NumPy arrays
    input_sequences = np.array(input_sequences)
    targets = np.array(targets)
    return input_sequences, targets

# Main function which combines all relevant steps for data preparation.
# Inputs:
#       - X: input features
#       - y: target variable
#       - test size: test size expressed as the percentage of total size of data
#       - test start date: if None ignore (default setting), if not None then split the data into train and test sets on this date
#       - do segmentation: if True, CPD is used to detect change points, data is divided into segments and scaled based on segment-wise statistics; 
#                          if False, scaling is done on entire time series at once
# Outputs:
#       - X train: input variables for the train set
#       - y train: target variable for the train set
#       - X test: input variables for the test set
#       - y test: target variable for the test set
# def prepare_input_dataset(X, y, test_size = 0.2, do_segmentation = False, window_size=7, step_size=1):

#     # Split the data into train and test
#     X_train, X_test = get_train_test_split(X, test_size)
#     y_train, y_test = get_train_test_split(y, test_size)

#     # Perform scaling on input variables, target variable does not need to be scaled
#     if do_segmentation:
#         segment_ind = scaling.get_segment_indexes(test_size)
#         X_train_sc, X_test_sc = scaling.scale_per_segment(X_train, X_test, segment_ind)
#     else:
#         scaler = scaling.TanhScaler()
#         X_train_sc, X_test_sc

#     # Convert data into form appropriate for LSTM using sliding window technique
#     X_train, y_train = apply_sliding_window(X_train_sc, y_train, window_size, step_size)
#     X_test, y_test = apply_sliding_window(X_test_sc, y_test, window_size, step_size)

#     return X_train, y_train, X_test, y_test

# Load the data into an instance of the custom made class CustomLSTMDataset and divide the data into batches.
def make_data_loader(X, y, batch_size = 10):
    dataset = CustomLSTMDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

def prepare_input_data(X, y, test_size, window_size, step_size, do_segmentation=False):
    # Split the data into train and test
    X_train_og, X_test_og = get_train_test_split(X, test_size)
    y_train_og, y_test_og = get_train_test_split(y, test_size)
    
    # Perform scaling on input variables, target variable does not need to be scaled
    scaler = None
    if do_segmentation:
        segment_ind = scaling.get_segment_indexes(test_size)
        scaler = StandardScaler()
        X_train_sc, X_test_sc = scaling.scale_per_segment(X_train_og, X_test_og, segment_ind, scaler)
        y_train_sc, y_test_sc = scaling.scale_per_segment(y_train_og, y_test_og, segment_ind, scaler)
    else:
        scaler = StandardScaler()
        X_train_sc, X_test_sc = scaling.scale_train_and_test(X_train_og, X_test_og, scaler)
        y_train_sc, y_test_sc = scaling.scale_train_and_test(y_train_og, y_test_og, scaler)

    # Convert data into form appropriate for LSTM using sliding window technique
    X_train, y_train = apply_sliding_window(X_train_sc, y_train_sc, window_size, step_size)
    X_test, y_test = apply_sliding_window(X_test_sc, y_test_sc, window_size, step_size)

    return X_train, y_train, X_test, y_test, scaler

