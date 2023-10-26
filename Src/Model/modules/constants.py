import pandas as pd

# Define date range
start_date_str = "2018-08-02"
end_date_str = "2023-06-27"

start_date = pd.to_datetime(start_date_str, format="%Y-%m-%d")
end_date = pd.to_datetime(end_date_str, format="%Y-%m-%d")

date_range = pd.date_range(start_date, end_date, freq="D")

test_start_date = pd.to_datetime("2022-01-18")
test_start_date_casper = pd.to_datetime("2022-01-18")

# Define folder paths
root_dir = "/mnt/Ivana/"
data_dir = root_dir + "Data/"
temp_dir = data_dir +"Temp/"
tezos_dir = data_dir + "Tezos/"

tezos_results_dir = root_dir + "Results/Tezos/"
casper_results_dir = root_dir + "Results/Casper/"

model_input_dir = tezos_dir + "ModelInput/" # the final features which are used as inputs to the model are stored here
final_processed_data_dir = tezos_dir + "DataDuringProcessing/CleanDataShortTimePeriod/"

input_X_dir = model_input_dir + "X/"
input_y_dir = model_input_dir + "y/"

# Define file paths
# segmentation_file = tezos_dir + "CorrelationAndSegmentation/Tezos_segmentation_details.csv"
segmentation_file = data_dir +  "Casper/CorrelationAndSegmentation/Casper_segmentation_details.csv"

