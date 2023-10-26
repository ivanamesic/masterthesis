import pandas as pd
import json
import os
from os import listdir, getcwd
import requests
import multiprocessing as mp
import numpy as np


url_acc_count = "https://api.tzkt.io/v1/accounts/count"
url_accs = "https://api.tzkt.io/v1/accounts"

dest_dir = getcwd() + "/Data/Temp/Accounts/"
tgt_file = getcwd() + "/Data/Temp/Accounts_all.csv"

LIMIT = 10000

def get_data(offsets, index):
    params = {
        "limit" : LIMIT
    }
   
    result = []
    for o in offsets:
        params["offset"] = o
        response = json.loads(requests.get(url= url_accs, params=params).text)
        result.extend(response)

    df = pd.DataFrame.from_dict(result)

    file_name = dest_dir + "Data" + str(index) + ".csv"
    print(len(df), file_name)

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

    acc_count = int(requests.get(url = url_acc_count, params = params).text)

    offsets = range(0, acc_count, LIMIT)
    n_cpu = mp.cpu_count() 

    if len(offsets) < n_cpu:
        n_cpu = len(offsets)
    
    for i, chunk in enumerate(np.array_split(offsets, n_cpu-1)):
       
        proc = mp.Process(target = get_data, args = (chunk, i, ))
        procs.append(proc)
        proc.start()
    for proc in procs:
      proc.join()

    join_files(src_dir=dest_dir, tgt_file=tgt_file)