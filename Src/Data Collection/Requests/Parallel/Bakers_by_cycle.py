import pandas as pd
import json
import os
from os import listdir, getcwd
import requests
import multiprocessing as mp
import numpy as np

url_baking_rewards = "https://api.tzkt.io/v1/rewards/bakers/"

address_file = "/mnt/Ivana/Data/Temp/delegate_addresses.json"
dest_dir = "/mnt/Ivana/Data/Temp/Baking/"
tgt_file =  "/mnt/Ivana/Data/Tezos/Raw_data/Baking_data.csv"

LIMIT = 10000

def get_data(addresses, index):
    params = {
        "limit" : LIMIT
    }
   
    result = []
    for a in addresses:
        response = json.loads(requests.get(url= url_baking_rewards + a, params=params).text)
        for r in response:
            r["address"] = a
        result.extend(response)

    df = pd.DataFrame.from_dict(result)

    file_name = dest_dir + "Data" + str(index) + ".csv"
    df.to_csv(file_name, index=False)
      
def join_files(src_dir, tgt_file):
    all_dfs = []

    for file in listdir(src_dir):
        df = pd.read_csv(src_dir + file, low_memory=False)
        all_dfs.append(df)

    result = pd.concat(all_dfs, axis=0) 
    result.to_csv(tgt_file, index=False)

def clean_directory(dir_path):
    for file in listdir(dir_path):
        os.remove(dir_path + file)

if __name__ == "__main__":
    procs = []

    params = {
        "limit" : LIMIT
    }

    clean_directory(dest_dir)

    with open(address_file, "r") as f:
        addresses = json.load(f)

    n_cpu = mp.cpu_count() 
    
    for i, chunk in enumerate(np.array_split(addresses, n_cpu-1)):
       
        proc = mp.Process(target = get_data, args = (chunk, i, ))
        procs.append(proc)
        proc.start()
    for proc in procs:
      proc.join()

    join_files(src_dir=dest_dir, tgt_file=tgt_file)

    clean_directory(dest_dir)