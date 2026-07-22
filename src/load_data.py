import os
import argparse
from get_data import read_params, get_data


def load_and_save(config_path):
    config = read_params(config_path)

    
    df = get_data(config_path)

   
    df = df.rename(columns={"quality": "TARGET"})

    
    raw_data_path = config["load_data"]["raw_dataset_csv"]

   
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

   
    df.to_csv(raw_data_path, sep=",", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    load_and_save(config_path=parsed_args.config)