import requests
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import os
import NFT_functions as NFTfunc

token_transfer_url  = "https://api.tzkt.io/v1/tokens/transfers"
token_transfer_count_url = "https://api.tzkt.io/v1/tokens/transfers/count"

LIMIT = 10000

nft_dir = "/mnt/Ivana/Data/Tezos/DataDuringProcessing/NFT/"
accounts_dir = nft_dir + "Accounts/"


nft_ids = {
    "Generative Springtime 02": [730111, "KT1RJ6PbjHpwc3M5rw5s2Nbmefwbuwbdxton"],
    "Garden, Monoliths #89" : [146074, "KT1KEa8z6vWXDJrVqtMrAeDVzsvxat3kHaCE"],
    "Window Still Life 001": [2371, "KT1RJ6PbjHpwc3M5rw5s2Nbmefwbuwbdxton"],
    "Window Still Life 101": [763422, "KT1RJ6PbjHpwc3M5rw5s2Nbmefwbuwbdxton"],
    "Tezzardz #514": [514, "KT1LHHLso8zQWQWg1HUukajdxxbkGfNoHjh6"],
    "Tezzardz #520" : [520, "KT1LHHLso8zQWQWg1HUukajdxxbkGfNoHjh6"],
    "Birth iii": [ 107012, "KT1RJ6PbjHpwc3M5rw5s2Nbmefwbuwbdxton"],
    "Generative Zlatna i" : [158552, "KT1RJ6PbjHpwc3M5rw5s2Nbmefwbuwbdxton"]
} 

artists = {
    "zancan.tez" : ["Generative Springtime 02", "Garden, Monoliths #89"],
    "jjjjjohn": ["Window Still Life 001", "Window Still Life 101"],
    "Tezzards": ["Tezzards #514", "Tezzards #520"],
    "Iskra Velitchkova": ["Birth iii", "Birth iii - Alive"]
}



accounts = {
    "zancan.tez": "tz1gBXG9fg8RMDH69KfKqwoTH5sFDmzt5yzm",
    "jjjjjohn" : "tz1gqaKjfQBhUMCE6LhbkpuittRiWv5Z6w38",
    "Iskra Velitchkova" : "tz1gVKxpEGC7QW1fZyEEMEW2kgJRbpgWLNpD",
    "Tezzardz": "tz1Nfr2qhuoGJSmK4oGv93QVfPgzTKChfcNK",
    "yazid.tez" : "tz1QgjmhrUD3X7kgS9mMHbUz4cS6uDiFGhAU"
}

# for acc_name, acc_address in accounts.items():
#     df_acc = NFTfunc.request_all_nfts_from_account(acc_address)
#     df_acc["accountName"] = acc_name
#     df_acc["accountAddress"] = acc_address

#     df_acc.to_csv(accounts_dir + acc_name + ".csv", index=False)


def call_token_transfer_history_creation(x, tgt_dir):
    token_contract, token_id = x["contract_address"], x["tokenId"]
    file_name = tgt_dir + token_contract + "_" + str(token_id) + ".csv"
    if os.path.exists(file_name):
        return
    return NFTfunc.request_token_transfers(token_id=token_id, token_contract=token_contract, transfer_file=file_name)

def merge_all_csv_files_from_folder(folder_path):
    all_dfs = []
    
    for file_name in os.listdir(folder_path):
        df = pd.read_csv(folder_path + file_name)
        all_dfs.append(df)
    
    result = pd.concat(all_dfs, axis=0)
    return result


temp_dir = nft_dir + "Temp/"
transfers_dir = nft_dir + "Transfers/"


print("starting requests")

for acc_file in tqdm(os.listdir(accounts_dir)):
    acc_name = acc_file[:-4]

    folder_name = temp_dir + acc_name + "/"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # file_name_result = transfers_dir + acc_name + "_all_transfers.csv"
    # if os.path.exists(file_name_result): continue

    df = pd.read_csv(accounts_dir + acc_file)
    df = df[(df.tokenId != "0") & (df.transfersCount > 1)]

    df["file_name"] = df.apply(lambda x: x["contract_address"] +"_" +str(x["tokenId"]) + ".csv", axis = 1)
    files = os.listdir(folder_name)
    df = df[~df.file_name.isin(files)].sort_values(by = "transfersCount").reset_index(drop=True)

    print(acc_name, " tokens with more than 1 transfer ", df.shape[0])

    # all_responses = df.apply(lambda x: call_token_transfer_history_creation(x, folder_name), axis = 1)
    tokens_to_request = [(x, y, z) for x, y, z in zip(df.contract_address.values, df.tokenId.values, df.transfersCount.values)]
    
    for contract_addr, token_id, transfers_count in tqdm(tokens_to_request):
        # print(contract_addr, token_id, transfers_count)
        if contract_addr == "KT1BRhcRAdLia3XQT1mPSofHyrmYpRddgj3s" and token_id == 6: continue
        file_name = folder_name + contract_addr + "_" + str(token_id) + ".csv"
        if os.path.exists(file_name): continue
        try:
            NFTfunc.request_token_transfers(token_id=token_id, token_contract=contract_addr, transfer_file=file_name)
        except:
            print("NOT SUCCEEDED: " + contract_addr + " " + token_id)       
    result = merge_all_csv_files_from_folder(folder_name)
    if "file_name" in result.columns:
        result.drop("file_name", axis = 1, inplace=True)

    # result.to_csv(file_name_result, index=False)
    
# df.

# %%
# details_list = []
# for name, values in nft_ids.items():
#     file_name = nft_dir + name + ".csv"

#     # if os.path.exists(file_name): continue
#     id, contract = values
#     token_details = request_token_transfers(id, contract, file_name)
#     details_list.append(token_details)

# token_details_df = pd.DataFrame.from_dict(details_list)
# token_details_df.to_csv(nft_dir + "Token_details.csv", index = False)


