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
