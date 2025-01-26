import unittest
import pandas as pd
import os
import sys
import json
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# If "CONF_PATH" isn't set, you can provide a fallback or handle it gracefully
CONF_FILE = os.getenv("CONF_PATH", "settings.json")

from training.train import DataProcessor, Training


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the config file
        with open(CONF_FILE, "r") as file:
            cls.conf = json.load(file)

        cls.data_dir = cls.conf["general"]["data_dir"]
        cls.train_path = os.path.join(cls.data_dir, cls.conf["train"]["table_name"])

    def test_data_extraction(self):
        """
        Test that data_extraction returns a pandas DataFrame
        from the configured training CSV path.
        """
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "DataFrame is unexpectedly empty.")

    def test_prepare_data(self):
        """
        Test that prepare_data can load and (optionally) sample the dataset.
        Here we request 100 rows and expect exactly 100 in return,
        assuming the dataset has at least that many rows.
        """
        dp = DataProcessor()
        df = dp.prepare_data(100)
        # If the underlying CSV has fewer than 100 rows, your code won't sample.
        # The Iris dataset typically has 105 rows in 'train' if you used test_size=0.3
        # in data_generation. This test will pass if the dataset is large enough.
        self.assertEqual(
            df.shape[0], 
            100, 
            f"Expected 100 rows, got {df.shape[0]}. " 
            "Ensure your train.csv has at least 100 rows."
        )


class TestTraining(unittest.TestCase):
    def test_train(self):
        """
        Test a simple training routine on a mock dataset.
        Verifies that the model is a PyTorch nn.Module,
        and that a forward pass is possible after training.
        """
        tr = Training()
        # Create a minimal DataFrame with the same columns used in training
        X_train = pd.DataFrame({
            "sepal length (cm)": [5.1, 4.9, 4.7, 4.6],
            "sepal width (cm)": [3.5, 3.0, 3.2, 3.1],
            "petal length (cm)": [1.4, 1.4, 1.3, 1.5],
            "petal width (cm)": [0.2, 0.2, 0.2, 0.2],
        })
        y_train = pd.Series([0, 1, 2, 1], dtype=int)

        # Train the model briefly
        tr.train(X_train, y_train)
        
        # Check that model is a PyTorch module
        self.assertTrue(isinstance(tr.model, torch.nn.Module))

        # Attempt a forward pass
        with torch.no_grad():
            inputs = torch.tensor(X_train.values, dtype=torch.float32)
            outputs = tr.model(inputs)
        
        # Should output logits of shape (4, 3)
        self.assertEqual(outputs.shape, (4, 3))


if __name__ == '__main__':
    unittest.main()
