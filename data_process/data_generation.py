"""
Data Generation Script for Iris Dataset

This script loads the Iris dataset and splits it into training and inference sets,
then saves them as CSV files locally based on the configuration provided.
"""

import os
import sys
import json
import logging
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Check environment vs. local file
env_conf_path = os.getenv("CONF_PATH")
if not env_conf_path:
    logger.warning("ENV 'CONF_PATH' not set, falling back to local 'settings.json'.")
    env_conf_path = "settings.json"

# Now verify that this file actually exists
if not os.path.exists(env_conf_path):
    logger.error(f"Configuration file not found at {env_conf_path}.")
    sys.exit(1)

try:
    with open(env_conf_path, "r") as file:
        conf = json.load(file)
    logger.info("Configuration settings loaded successfully.")
except json.JSONDecodeError as e:
    logger.error(f"Error decoding JSON from {env_conf_path}: {e}")
    sys.exit(1)

# Validate configuration
required_keys = ["general", "train", "inference"]
for key in required_keys:
    if key not in conf:
        logger.error(f"Missing '{key}' section in configuration.")
        sys.exit(1)

# Define paths from configuration
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])
INFERENCE_PATH = os.path.join(DATA_DIR, conf["inference"]["inp_table_name"])

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
logger.info(f"Data directory set to: {DATA_DIR}")


def main():
    """Main function to load the Iris dataset, split it, and save as CSV files."""
    try:
        logger.info("Starting data generation script.")

        # Load data from sklearn.datasets
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df["target"] = iris.target
        logger.info("Iris dataset loaded successfully.")

        # Split into training and inference sets
        test_size = conf["train"].get("test_size", 0.2)
        random_state = conf["general"].get("random_state", 42)
        train, inference = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        logger.info(
            f"Data split into training ({train.shape}) and inference ({inference.shape}) sets."
        )

        # Drop the target column for the inference dataset
        inference_no_label = inference.drop(columns=["target"])

        # Save to CSV files
        train.to_csv(TRAIN_PATH, index=False)
        inference_no_label.to_csv(INFERENCE_PATH, index=False)
        logger.info(f"Training data saved to {TRAIN_PATH}.")
        logger.info(f"Inference data (no labels) saved to {INFERENCE_PATH}.")

        logger.info("Data generation script completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during data generation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
