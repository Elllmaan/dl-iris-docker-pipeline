"""
Script loads a trained model and data for inference, and predicts results.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load configuration
CONF_FILE = os.getenv("CONF_PATH")

# Fallback to default 'settings.json' if CONF_PATH is not set
if not CONF_FILE:
    logger.warning("ENV 'CONF_PATH' not set, falling back to local 'settings.json'.")
    CONF_FILE = os.path.join(os.path.dirname(ROOT_DIR), "settings.json")

# Verify that the configuration file exists
if not os.path.exists(CONF_FILE):
    logger.error(f"Configuration file not found at {CONF_FILE}.")
    sys.exit(1)

try:
    with open(CONF_FILE, "r") as file:
        conf = json.load(file)
    logger.info("Configuration settings loaded successfully.")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from {CONF_FILE}: {e}")
    sys.exit(1)

# Define paths from configuration
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
RESULTS_DIR = get_project_dir(conf["general"]["results_dir"])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--infer_file",
    help="Specify inference data file",
    default=conf["inference"]["inp_table_name"],
)
parser.add_argument(
    "--out_path",
    help="Specify the path to the output CSV file with predictions",
    default=None
)

def get_latest_model_path() -> str:
    """
    Gets the path of the latest saved .pth model by checking filenames
    that match the datetime format from config.
    """
    latest = None
    latest_time = None
    for dirpath, dirnames, filenames in os.walk(MODEL_DIR):
        for filename in filenames:
            if filename.endswith(".pth"):
                try:
                    file_time = datetime.strptime(
                        filename, conf["general"]["datetime_format"] + ".pth"
                    )
                    if latest_time is None or file_time > latest_time:
                        latest_time = file_time
                        latest = filename
                except ValueError:
                    logging.warning(
                        f"Filename {filename} does not match the datetime format."
                    )
    if latest is None:
        logging.error("No valid model files found.")
        sys.exit(1)
    return os.path.join(MODEL_DIR, latest)

def load_pytorch_model(path: str) -> nn.Module:
    """
    Loads and returns the PyTorch model from the given .pth file.
    """
    try:
        if not os.path.exists(path):
            logging.error(f"Model file not found: {path}")
            sys.exit(1)

        logging.info(f"Loading PyTorch model from: {path}")

        # Define the same architecture used in training
        model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model.eval()
        logging.info("Successfully loaded PyTorch model.")
        return model

    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        sys.exit(1)

def get_inference_data(path: str) -> pd.DataFrame:
    """
    Loads and returns data for inference from the specified CSV file.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Inference data columns: {df.columns.tolist()}")
        logging.info(f"Inference data shape: {df.shape}")
        if df.empty:
            logging.warning("Inference CSV is empty, results might be invalid.")
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)

def predict_results(model: nn.Module, infer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the results using the given PyTorch model and attach them
    to the original DataFrame in a 'results' column.
    """
    try:
        feature_columns = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]

        # Check if all required feature columns are present
        missing_cols = [col for col in feature_columns if col not in infer_data.columns]
        if missing_cols:
            logging.error(
                f"Missing required feature columns for inference: {missing_cols}"
            )
            sys.exit(1)

        selected_data = infer_data[feature_columns]
        logging.info(
            f"Selected feature columns for prediction: {selected_data.columns.tolist()}"
        )

        infer_tensor = torch.tensor(selected_data.values, dtype=torch.float32)
        infer_dataset = TensorDataset(infer_tensor)
        dataloader = DataLoader(infer_dataset, batch_size=16, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0]
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.numpy())

        infer_data["results"] = predictions
        return infer_data
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        sys.exit(1)

def store_results(results: pd.DataFrame, path: str = None) -> None:
    """
    Stores the prediction results to a CSV file. If no path is provided,
    a timestamp-based filename is created in the results directory.
    """
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf["general"]["datetime_format"]) + ".csv"
        path = os.path.join(RESULTS_DIR, path)
    results.to_csv(path, index=False)
    logging.info(f"Results saved to {path}")

def main():
    """
    Main function that handles model loading (either latest or specific),
    data loading, prediction, and storing results.
    """
    args = parser.parse_args()

    # Decide whether to use the latest model or a specific path
    use_latest = conf["inference"].get("use_latest_model", True)
    if use_latest:
        model_path = get_latest_model_path()
    else:
        model_path = os.path.join(MODEL_DIR, conf["inference"]["model_name"])
        if not os.path.exists(model_path):
            logging.error(f"Specified model not found at {model_path}.")
            sys.exit(1)

    model = load_pytorch_model(model_path)

    infer_file_path = os.path.join(DATA_DIR, args.infer_file)
    infer_data = get_inference_data(infer_file_path)
    results = predict_results(model, infer_data)

    store_results(results, args.out_path)
    logging.info(f"Prediction results:\n{results}")

if __name__ == "__main__":
    main()