import pandas as pd
import numpy as np
import modules.constants as const

# A custom scaler class which implements the tanh scaling function.
class TanhScaler():
    def __init__(self):
        self.mean = None
        self.std = None

    # Calculate the mean and standard deviation along each feature. These are the hyperparameters of the scaler
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    # Scale the data using the precomputed mean and standard deviation and applying tanh scaling
    def transform(self, data):
        scaled_data = 0.5*(np.tanh(0.01 * (data - self.mean) / self.std) + 1)
        return scaled_data

    # First fit the data to the scaler and then transform it
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    # Reverse the scaling using the fitted mean and std
    def inverse_transform(self, scaled_data):
        unscaled_data = self.mean + (self.std * np.arctanh(2 * scaled_data - 1)) / 0.01
        return unscaled_data



# Scale the data segment-wise. Segment-wise scaling is done only for train set, since the scaler should not be fitted on test data. The test data is transformed using the scaler which was fitted on the last segment of the train set.
# Inputs:
#       - data train: train input data
#       - data test: test input data
#       - train segment indexes: indexes which divide the training dataset into segments based on performed CPD
# Output:
#       - scaled train set, scaled test set
def scale_per_segment(data_train, data_test, train_segment_indexes, scaler):
    train_scaled = np.empty((0, data_train.shape[1]))
    for start_index, end_index in train_segment_indexes:
        segment = data_train[start_index:end_index, ]

        if end_index == -1:
            segment = data_train[start_index:, ]
        
        scaled_segment = scaler.fit_transform(segment)

        train_scaled = np.concatenate((train_scaled, scaled_segment), axis = 0)
    
    test_scaled = scaler.transform(data_test)
    return train_scaled, test_scaled





# Function which loads the data frame containing results of the CPD analysis and returns indexes which define segments. It returns only segments which are contained within the train data set.
# Input:
#       - test_size: size of the test set; this also defines the endpoint of the train set
def get_segment_indexes(test_size):
    data_len = len(const.date_range)
    cutoff_index = int(data_len*(1-test_size))

    segment_df = pd.read_csv(const.segmentation_file)
    
    train_segments = []
    
    for i, row in segment_df.iterrows():
        start_index = row["Start Index"]
        end_index = row["End Index"]

        if start_index >= cutoff_index: break

        if end_index > cutoff_index:
            end_index = cutoff_index
        
        train_segments.append((start_index, end_index))

    return train_segments


def scale_train_and_test(train, test, scaler):
    train_sc = scaler.fit_transform(train)
    test_sc = scaler.transform(test)

    return train_sc, test_sc

# def scale_without_segmentation(data_train, data_test):
#     if (not isinstance(data_train, np.ndarray)) or (not isinstance(data_test, np.ndarray)):
#         raise TypeError("Scaling is performed on numpy arrays and not data frames")
    
#     scaler = StandardScaler()
#     train_scaled = scaler.fit_transform(data_train)
#     test_scaled = scaler.transform()
    
#     return train_scaled, test_scaled


# def tanh_scaling(v, mean, std):
#     scaled_data = 0.5*(np.tanh(0.01 * (v - mean) / std) + 1)
#     return scaled_data

# def inverse_tanh_scaling(scaled_data, mean, std):
#     unscaled_data = mean + std * np.arctanh(2 * scaled_data - 1) / 0.01
#     return unscaled_data

# # Scale each variable of the input data using tanh scaling, but without segmentation. The scaler should be fitted on training data and transformed on test data.
# # Test data should not be included in fitting the model. Input data should be numpy arrays.
# # Inputs: train and test sets
# # Output: scaled training and test sets
# def scale_input_data_tanh(X_train, X_test):
#     # train_mean = np.mean(X_train, axis = 0)
#     # train_std = np.mean(X_train, axis = 0)
#     tanh_scaler = TanhScaler()

#     X_train_scaled = tanh_scaler.fit_transform(X_train)
#     X_test_scaled = tanh_scaler.transform(X_test)

#     return X_train_scaled, X_test_scaled