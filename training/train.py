"""
This script prepares the data and runs the training, then saves the model.
"""

import argparse
import os
import sys
import json
import logging
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Comment these lines if you have problems with MLFlow installation
import mlflow
import mlflow.pytorch

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv("CONF_PATH")

from utils import get_project_dir, configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Check environment vs. local file
if not CONF_FILE:
    logger.warning("ENV 'CONF_PATH' not set, falling back to local 'settings.json'.")
    CONF_FILE = os.path.join(ROOT_DIR, "..", "settings.json")

# Now verify that this file actually exists
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

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])

# Initialize parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_file",
    help="Specify training data file",
    default=conf["train"]["table_name"],
)
parser.add_argument("--model_path", help="Specify the path for the output model")


class DataProcessor:
    """
    A class to handle data extraction and random sampling for training.
    """

    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        """
        Loads the training CSV, randomly samples rows if max_rows is specified,
        and returns the resulting DataFrame.
        """
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        # Handle no data scenario
        if df.empty:
            logging.error("Loaded training data is empty. Exiting.")
            sys.exit(1)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        """Loads data from a CSV file."""
        logging.info(f"Loading data from {path}...")
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            sys.exit(1)

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        """Randomly sample the DataFrame if max_rows is defined and feasible."""
        if not max_rows or max_rows < 0:
            logging.info("Max_rows not defined. Skipping sampling.")
        elif len(df) < max_rows:
            logging.info("Size of DataFrame is less than max_rows. Skipping sampling.")
        else:
            df = df.sample(
                n=max_rows, replace=False, random_state=conf["general"]["random_state"]
            )
            logging.info(f"Random sampling performed. Sample size: {max_rows}")
        return df


class Training:
    """
    A class encapsulating the PyTorch model, its training, testing, and saving logic.
    """

    def __init__(self) -> None:
        # Simple feed-forward neural net
        self.model = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 3),
        )

    def run_training(
        self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33
    ) -> None:
        """
        Main routine to split data, train the model, log training duration,
        compute metrics, and save the trained model.
        """
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        training_duration = end_time - start_time
        logging.info(f"Training completed in {training_duration} seconds.")
        mlflow.log_metric("training_duration_seconds", training_duration)

        f1 = self.test(X_test, y_test)
        mlflow.log_metric("f1_score", f1)

        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        """
        Splits DataFrame into train/test sets for features and target.
        """
        logging.info("Splitting data into training and test sets...")
        return train_test_split(
            df[
                [
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ]
            ],
            df["target"],
            test_size=test_size,
            random_state=conf["general"]["random_state"],
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """
        Trains the model on the given data using parameters read from settings.json.
        """
        logging.info("Training the model...")

        batch_size = conf["train"].get("batch_size", 16)
        learning_rate = conf["train"].get("learning_rate", 0.001)
        epochs = conf["train"].get("epochs", 20)

        # Log training parameters from config
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(dataloader)
            logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
            mlflow.log_metric("loss", epoch_loss, step=epoch + 1)

    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        """
        Evaluates the trained model on the test set, returning the F1 score.
        """
        logging.info("Testing the model...")

        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, y_pred = torch.max(outputs, 1)

        res = f1_score(y_test_tensor.numpy(), y_pred.numpy(), average="weighted")
        logging.info(f"f1_score: {res}")
        return res

    def save(self, path: str) -> None:
        """
        Saves the trained model weights in .pth format to the models directory.
        Also logs the model to MLflow.
        """
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(
                MODEL_DIR,
                datetime.now().strftime(conf["general"]["datetime_format"]) + ".pth",
            )
        else:
            path = os.path.join(MODEL_DIR, path)

        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")

        # Log the model as an artifact
        mlflow.pytorch.log_model(self.model, "model")
        logging.info("Model logged to MLflow.")


def main():
    """
    Main function to configure logging, parse args, prepare data, and start MLflow run.
    """
    configure_logging()
    args = parser.parse_args()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf["train"]["data_sample"])

    with mlflow.start_run():
        # Log configuration parameters
        mlflow.log_params(conf["train"])

        tr.run_training(df, test_size=conf["train"]["test_size"])

        script_path = os.path.abspath(__file__)
        mlflow.log_artifact(script_path, "training_script")

        logging.info("MLflow run completed.")


if __name__ == "__main__":
    main()
