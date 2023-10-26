import pandas as pd
import json
import os
from os import listdir, getcwd
import requests
import multiprocessing as mp
import numpy as np
import time

url_baking_rewards = "https://api.tzkt.io/v1/accounts/"

address_file = "/mnt/Ivana/Data/Temp/delegate_addresses.json"

dest_dir = "/mnt/Ivana/Data/Temp/Baking/"
tgt_file =  "/mnt/Ivana/Data/Tezos/Raw_data/Bakers_balances.csv"

LIMIT = 10000

def get_data(address):
    params = {
        "limit" : LIMIT
    }
   
    result = []
    i = 0

    ret = False
    while(True):
        params["offset"] = i*LIMIT
        try:
            response = json.loads(requests.get(url= url_baking_rewards + address + "/balance_history", params=params).text)
        
        except Exception as error:
            ret = True
            print(error)
            break
        if len(response) == 0: break
        result.extend(response)
        i+=1

        if len(response) < LIMIT: break

    if ret: 
        return

    df = pd.DataFrame.from_dict(result)
    df["address"] = address

    file_name = dest_dir + address  + ".csv"
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

    params = {
        "limit" : LIMIT
    }

    #clean_directory(dest_dir)

    done_addrs = [i.replace(".csv", "") for i in os.listdir(dest_dir)]

    with open(address_file, "r") as f:
        all_bakers = json.load(f)

    n_cpu = 15
    
    start_t = time.time()

    addresses = [i  for i in all_bakers if i not in done_addrs]

    for i in range(0, len(addresses), n_cpu):
        addr_chunk = addresses[i:i+n_cpu]
        print(i)
        procs = []

        for a in addr_chunk:
            proc = mp.Process(target = get_data, args = (a,  ))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    join_files(src_dir=dest_dir, tgt_file=tgt_file)

    #clean_directory(dest_dir)