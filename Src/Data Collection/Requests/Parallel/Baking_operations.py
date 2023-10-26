import pandas as pd
import json
import os
from os import listdir, getcwd
import requests
import multiprocessing as mp
import numpy as np
from tools import *
from tqdm import tqdm


url_baking_rewards = "https://api.tzkt.io/v1/operations/baking"

dest_dir = "/mnt/Ivana/Data/Temp/Baking/"
tgt_file =  "/mnt/Ivana/Data/Tezos/Raw_data/Baking_data.csv"

END_BLOCK = 3782050

LIMIT = 10000

def get_data(blocks, index):
     
    result = []
    for b in tqdm(blocks):
        params = { "limit" : LIMIT, "level.ge" : b,  "level.lt" : b + LIMIT }

        try:   
            resp = requests.get(url= url_baking_rewards , params=params)
            response = json.loads(resp.text)
            result.extend(response)
        except:
            print(resp.reason)

    result_norm = normalize_json_list(result)
    df = pd.DataFrame.from_dict(result_norm)

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

    blocks = range(0, END_BLOCK, LIMIT)

    n_cpu = mp.cpu_count() 
    
    for i, chunk in enumerate(np.array_split(blocks, n_cpu-1)):
       
        proc = mp.Process(target = get_data, args = (chunk, i, ))
        procs.append(proc)
        proc.start()
    for proc in procs:
      proc.join()

    join_files(src_dir=dest_dir, tgt_file=tgt_file)

    clean_directory(dest_dir)