import requests, json
import pandas as pd


LIMIT = 10000

token_transfer_url  = "https://api.tzkt.io/v1/tokens/transfers"
token_transfer_count_url = "https://api.tzkt.io/v1/tokens/transfers/count"
transaction_details_url = "https://api.tzkt.io/v1/operations/transactions"

# Parse json response containing details about 1 token transfer
def parse_token_transfer(js):
    result = {
        "id": js["id"], 
        "timestamp": js["timestamp"],
        "amount": js["amount"],
        "transactionId": js["transactionId"]
    }

    if "from" in js.keys():
        result["fromAddress"] = js["from"]["address"]
        if "alias" in js["from"].keys():
            result["fromAlias"] = js["from"]["alias"]
    
    if "to" in js.keys(): 
        result["toAddress"] = js["to"]["address"]
        if "alias" in js["to"].keys():
            result["toAlias"] = js["to"]["alias"]

    return result

# Parse token details about 1 token from metadata
def parse_token_details(js):
    token_js = js["token"]

    # if "metadata" not in js["token"].keys():
    #     print(json.dumps(js, indent=2))
    #     return
    
    creators = None
    if "creators" in js["token"]["metadata"].keys():
        creators = token_js["metadata"]["creators"]

    name = None
    if "name" in js["token"]["metadata"].keys():
        name = token_js["metadata"]["name"]

    result = {
        "name": name,
        "contract_address": token_js["contract"]["address"],
        "tokenId" : token_js["tokenId"],
        "supply" : token_js["totalSupply"],
        "creators" : creators,
        "transfersCount" : js["transfersCount"]
    }

    return result

# Request all FA2 protocol tokens for the given account address
def request_all_nfts_from_account(account_address, file_name=None):
    url = "https://api.tzkt.io/v1/tokens/balances"
    
    params = {
        "account": account_address,
        "token.standard":"fa2",
        "LIMIT" : 10000
    }

    resp = json.loads(requests.get(url=url, params=params).text)

    result = [parse_token_details(i) for i in resp if "metadata" in i["token"].keys()]
    df = pd.DataFrame.from_dict(result)

    if file_name is not None:
        df.to_csv(file_name, index = False)

    return df


# Request all token transfers for a token determined by an id and token contract and return in dataframe format
def request_token_transfers(token_id, token_contract, transfer_file = None):
    params = {
        "limit" : LIMIT,
        "token.tokenId": token_id,
        "token.contract": token_contract
    }

    nr_token_transfers = int(requests.get(url = token_transfer_count_url, params = params).text)
    
    result = []

    for i in range(0, nr_token_transfers, LIMIT):
        params["offset"] = i
        response = json.loads(requests.get(url = token_transfer_url, params = params).text)

        if len(response) == 0:
            break

        parsed = [parse_token_transfer(j) for j in response]
        result.extend(parsed)

    df = pd.DataFrame.from_dict(result)

    tgt_columns = ["id", "timestamp", "transactionId", "amount", "fromAddress", "fromAlias", "toAddress", "toAlias"]
    
    for col in tgt_columns:
        if col not in list(df.columns):
            df[col] = None

    df = df[tgt_columns]

    try:
        df[["hash", "type"]] = df.transactionId.apply(lambda x: get_transaction_hash_and_type_by_id(x))
        df[['price', "sender", 'sender_alias', 'target', 'target_alias']] = df.hash.apply(lambda x: get_transaction_details_by_hash(x))
    except:
        return pd.DataFrame()

    if transfer_file is not None:
        df.to_csv(transfer_file, index=False)
    
    return df

# Helper function which requests additional transaction details by transaction hash and returns the amount, sender address, sender alias, target address and alias
def get_transaction_details_by_hash(tx_hash):
    url  = "https://api.tzkt.io/v1/operations/" + tx_hash
    resp = json.loads(requests.get(url=url).text)

    first_tx = resp[0]
    amount = 0
    if "amount" in first_tx.keys():
        amount = first_tx["amount"] 

    
    sender, target, sender_alias, target_alias = None, None, None, None
    
    if "sender" in first_tx.keys():
        sender = first_tx["sender"]["address"]

        if "alias" in first_tx["sender"].keys():
            sender_alias = first_tx["sender"]["alias"]

    if "target" in first_tx.keys():
        target = first_tx["target"]["address"]
        if "alias" in first_tx["target"].keys():
            target_alias = first_tx["target"]["alias"]

    return pd.Series([amount, sender, sender_alias, target, target_alias], index=['amount', "sender", 'sender_alias', 'target', 'target_alias'])

# Helper function which requests transaction details by transaction id and returns transaction hash and type
def get_transaction_hash_and_type_by_id(tx_id):
    url = "https://api.tzkt.io/v1/operations/transactions"
    params = {
        "id": tx_id
    }

    resp = json.loads(requests.get(url=url, params=params).text)[0]

    return pd.Series([resp["hash"], resp["parameter"]["entrypoint"]], index=['hash', 'type'])
