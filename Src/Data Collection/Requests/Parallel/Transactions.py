import pandas as pd
import json
import os
from os import listdir, getcwd
import requests
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import time


url_transactions = "https://api.tzkt.io/v1/operations/transactions"

root_dir = "/mnt/Ivana/"
dest_dir = root_dir + "Data/Temp/Transactions/"

blocks_file = root_dir + "Data/Tezos/Processed/Blocks_by_month.csv"
tgt_file = root_dir + "/Data/Tezos/Raw_data/Transactions.csv"

LIMIT = 10000

def normalize_json(data: dict) -> dict:
  
    new_data = dict()
    for key, value in data.items():
        if not isinstance(value, dict):
            new_data[key] = value
        else:
            for k, v in value.items():
                new_data[key + "_" + k] = v
  
    return new_data

def normalize_json_list(json_list: list) -> list:
    result = []
    for j in json_list:
        result.append(normalize_json(j))

    return result 

def get_data(row):
    file_name = dest_dir + "Trans" + row["date"] + ".csv"

    result = []

    i = 0
    params = {
        "limit": LIMIT,
        "level.ge": row["startLevel"],
        "level.lt": row["endLevel"]
    }

    while(True):
        try:
            params["offset"] = i*LIMIT
            resp = requests.get(url= url_transactions, params=params)
            response = json.loads(resp.text)     

            if len(response) == 0: break       
            result.extend(response)
        except:
            print(resp.reason)
            return
        
        if len(response) < LIMIT: break
        i+=1


    if len(result) == 0: return
    result_norm = normalize_json_list(result)

    df = pd.DataFrame.from_dict(result_norm)
    cols = ['type', 'id', 'level', 'timestamp', 'block', 'hash', 'sender_address', 'gasLimit', 'gasUsed', 'storageLimit', 'storageUsed', \
       'bakerFee', 'storageFee', 'allocationFee', 'target_address', 'amount',  'status']
    df = df[cols]

    print("saving " + file_name)
    df.to_csv(file_name, index=False)
      
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

    #clean_directory(dest_dir)
    df_blocks = pd.read_csv(blocks_file)
    
    done_months = [file[:-4].replace("Trans", "") for file in os.listdir(dest_dir)]

    for j in range(0, df_blocks.shape[0], 5):
        chunk = df_blocks.loc[j:j+5, ]
        for i, row in tqdm(chunk.iterrows()):      
            if row["date"] in done_months: continue 
            proc = mp.Process(target = get_data, args = (row, ))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
        time.sleep(5)
    
    