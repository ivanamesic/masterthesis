import pandas as pd
import json
import os
from os import listdir, getcwd
import requests
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import time


url = "https://api.tzstats.com/tables/op"

root_dir = "/mnt/Ivana/"
dest_dir = root_dir + "Data/Temp/Transactions_tzstats/"

blocks_file = root_dir + "Data/Tezos/Processed/Blocks_by_month.csv"

tgt_file = root_dir + "/Data/Tezos/Raw_data/Transactions_tzstats.csv"

LIMIT = 10000



def get_data(startLevel, endLevel, file_name):
    transaction_columns = ["height", "time", "hash", "sender", "receiver", "type", "is_success", "volume"]

    all_dfs = []
    for block in tqdm(range(startLevel, endLevel, 100)):
        params= {
            "limit": 50000,
            "height.gte": block,
            "height.lt": block + 100,
            "type": "transaction",
            "columns": ",".join(transaction_columns)
        }

        response = requests.get(url = url, params = params)
        result = json.loads(response.text)
        df = pd.DataFrame.from_dict(result)
        all_dfs.append(df)
    
    result_df = pd.concat(all_dfs, axis = 0)
    result_df.columns = transaction_columns
    result_df.to_csv(file_name, index=False)

        


    
def join_files(src_dir, tgt_file):
    all_dfs = []

    for file in listdir(src_dir):
        df = pd.read_csv(src_dir + file, low_memory=False)
        all_dfs.append(df)

    if len(all_dfs) == 0: return
    result = pd.concat(all_dfs, axis=0) 
    result.to_csv(tgt_file, index=False)

def clean_directory(dir_path):
    for file in listdir(dir_path):
        os.remove(dir_path + file)

if __name__ == "__main__":
    procs = []


    transaction_columns = ["height", "time", "hash", "sender", "receiver", "type", "is_success", "volume"]
    #clean_directory(dest_dir)
    df_blocks = pd.read_csv(blocks_file)
    
    for j in range(0, df_blocks.shape[0], 10):
        chunk = df_blocks.loc[j:j+10, ]
        procs = []

        print(j)
        for i, row in chunk.iterrows():      
            file_name = dest_dir + "Tx_" + row["date"] + ".csv"

            proc = mp.Process(target = get_data, args = (row["startLevel"], row["endLevel"], file_name, ))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
    
    