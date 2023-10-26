import numpy as np
import pandas as pd
import re

# This function takes 1 arguments as input: an array containing 2 or more dataframes. 
# The function merges dataframes column-wise (axis 1) and returns the values of the merged array in numpy array format.
def merge_dataframes(df_arr):
    if not isinstance(df_arr, (list, tuple, np.ndarray)): 
        raise ValueError("The input to this function should be an array")
    
    all_values = []
    for df in df_arr:
        all_values.append(df.values)
    merged_array = np.hstack(all_values)
    return merged_array

# This function creates a custom dataset based on a number which represents a unique combination of input datasets.
# Inputs:
#       - num: integer number 0-31 which represents the combination of the input datasets. 
#               This number is turned into the binary version of itself and padded with leading zeros to reach length 5. 
#               Length 5 is needed since there are 5 input data frames, so every digit of the resulting number will serve as a flag for whether the input data set is included in the resulting data set.
#               So, if the digit on position i in the binary representation is 1, this means that the data frame with position i in the all_dfs list will be included in the resulting dataset.
#       - all dfs: array containing data frames with loaded input datasets
#       - data names: array containing names of input data frames in the same order that they are listed in all_dfs array
# Outputs:
#       - data: merged dataset with the unique combination of input data sets
#       - dict_chosen: a dictionary where keys are input data frames names and the values are the boolean flags of whether they were included in the result or not
def get_data_from_combination_number(num, all_dfs, data_names):
    binary_string = bin(num)[2:].zfill(len(all_dfs))
    arr = [int(c) for c in binary_string]
    chosen_dfs = [all_dfs[index] for index, value in enumerate(arr) if value == 1]
    data = merge_dataframes(chosen_dfs)

    dict_chosen = {"number":num}
    for i, data_name in enumerate(data_names):
        dict_chosen[data_name] = arr[i]
    return data, dict_chosen

def convert_list_of_tensors_to_array(x):
    res = []
    for i in x:
        res.append(x[0].numpy())
    return res

def convert_to_array_from_string(string_array):
    res = string_array.replace("]", "").replace("[", "").replace("\n", "")
    arr = res.split(" ")
    arr = np.array([float(i) for i in arr if i!=""])
    return arr

def convert_string_arrays_in_df(df):
    for i, row in df.iterrows():
        for j, col in enumerate(df.columns):
            data = row[col]
            if isinstance(data, str) and '[' in data:
                data = convert_to_array_from_string(data)
            df.at[i,str(col)] = data
    return df

def split_tuple_df_into_two(df_tuple):
    df_res1=df_tuple.copy()
    df_res2 = df_tuple.copy()

    for c in df_tuple.columns[1:]:
        df_res1[c] = df_tuple[c].apply(lambda x: x[0])
        df_res2[c] = df_tuple[c].apply(lambda x: x[1])
    
    return df_res1, df_res2

def convert_results_df_format(df_old):
    n_hidden = list(df_old.columns[1:])

    results = []
    for i, row in df_old.iterrows():
        for c in n_hidden:
            train_loss, val_loss = re.findall(r'\d+\.\d+', row[c]) 
            results.append({
                "hidden_neurons": c,
                "learning_rate": float(row["index"]),
                "train_loss": round(float(train_loss), 6),
                "validation_loss":round(float(val_loss), 6)
            })
    
    df = pd.DataFrame.from_dict(results)
    df = df.sort_values(by=["validation_loss"], ascending=True)
    return df