import os
import argparse
import pandas as pd
import numpy as np
import json
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

from get_data import read_params


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)

    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    model_dir = config["model_dir"]

    random_state = config["base"]["random_state"]

    target = [config["base"]["target_col"]]

    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["l1_ratio"]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    print("Train columns:", train.columns)
    print("Target from params:", target)
    print("Is target موجود؟", target[0] in train.columns)

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state
    )

    model.fit(train_x, train_y.values.ravel())

    preds = model.predict(test_x)

    rmse, mae, r2 = eval_metrics(test_y.values.ravel(), preds)

    print(f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio})")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")

    os.makedirs("report", exist_ok=True)

    with open(config["reports"]["scores"], "w") as f:
        json.dump({"rmse": rmse, "mae": mae, "r2": r2}, f, indent=4)

    with open(config["reports"]["params"], "w") as f:
        json.dump({"alpha": alpha, "l1_ratio": l1_ratio}, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    train_and_evaluate(config_path=args.config)