import yaml
import os
import json
import joblib
import numpy as np


BASE_DIR = os.path.dirname(__file__)
params_path = os.path.join(BASE_DIR, "..", "params.yaml")
schema_path = os.path.join(BASE_DIR, "schema_in.json")


class NotInRange(Exception):
    def __init__(self, message="Values entered are not in expected range"):
        self.message = message
        super().__init__(self.message)


class NotInCols(Exception):
    def __init__(self, message="Not in cols"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params()
    
    # 🔥 model path مظبوط
    model_dir_path = os.path.join(BASE_DIR, "..", config["webapp_model_dir"])
    
    print("MODEL PATH:", model_dir_path)
    print("EXISTS?", os.path.exists(model_dir_path))

    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]

    if 3 <= prediction <= 8:
        return prediction
    else:
        return "Unexpected result"


def get_schema():
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(dict_request):
    schema = get_schema()

    for col, val in dict_request.items():

        
        if col not in schema:
            raise NotInCols(f"{col} not in schema")

        
        try:
            value = float(val)
        except:
            raise NotInRange(f"{col} must be a number")

        if not (schema[col]["min"] <= value <= schema[col]["max"]):
            raise NotInRange(f"{col} out of range")

    return True


def form_response(dict_request):
    if validate_input(dict_request):
        data = [list(map(float, dict_request.values()))]
        response = predict(data)
        return response


def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(map(float, dict_request.values()))])
            response = predict(data)
            return {"response": response}

    except NotInRange as e:
        return {"error": str(e), "expected_range": get_schema()}

    except NotInCols as e:
        return {"error": str(e), "expected_cols": list(get_schema().keys())}

    except Exception as e:
        return {"error": str(e)}