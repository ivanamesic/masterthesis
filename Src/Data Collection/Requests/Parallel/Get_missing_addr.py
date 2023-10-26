import pandas as pd
import json

address_file = "/mnt/Ivana/Data/Temp/delegate_addresses.json"
balances_file =  "/mnt/Ivana/Data/Tezos/Raw_data/Bakers_balances.csv"

with open(address_file, "r") as f:
    all_bakers = json.load(f)

all_bakers = set(all_bakers)
df_balances = pd.read_csv(balances_file)

scraped_bakers = df_balances.address.unique().tolist()

get_addr = list(all_bakers.difference(set(scraped_bakers)))

with open("/mnt/Ivana/Data/Temp/get_bakers.json", "w") as f:
    json.dump(get_addr, f)