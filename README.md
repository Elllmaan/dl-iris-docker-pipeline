# MLE_basic_example

## Start

Choose between running the project locally or using Docker based on your preferences and setup.

### **Local Run**

Complete the following steps to run the project on your local machine.

#### 1.Clon the Repository
Fork the Repository:
Navigate to the ML_basic_example GitHub page.
Click the Fork button at the top right to create a copy under your GitHub account.
Clone the Repository Locally:

git clone https://github.com/Elllman/ML_basic_example.git
cd ML_basic_example

#### 2.Set Up the Development Environment

Create and activate a Python virtual environment to manage project dependencies separately from your global Python installation.

python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`

#### 3.Install Dependencies

Install all necessary Python packages listed in the requirements.txt file to ensure that the project has access to all required libraries and tools.

pip install -r requirements.txt

#### 4.Create a .env File

Set up environment variables by creating a .env file in the project's root directory. This file should specify the path to your configuration settings, ensuring that the project can access necessary configurations without hardcoding them.

echo "CONF_PATH=settings.json" > .env

#### 5.Generate Data

Generate the necessary datasets for training and inference.

python data_process/data_generation.py

This script will create the following CSV files in the data/ directory:

iris_train.csv
iris_inference.csv

#### 6.Train the Model

Execute the training script to develop your Machine Learning model. This step involves processing the training data, training the model, and saving the trained model to the designated directory for future use.

python training/train.py

#### 7.Run Inference

Use the trained model to make predictions on new, unseen data. Running the inference script will utilize the saved model to generate prediction results, which will be stored in the results/ directory for analysis.

python inference/run.py

### **Docker Run**

Utilize Docker to containerize the training and inference processes, ensuring consistency across different environments and simplifying dependency management.

#### 1.Clon the Repository
(Follow the same cloning steps as outlined in the Local Run section.)

#### 2.Build and Run the Training Docker Image
Use Docker to build an isolated environment for training the model.

docker build -f ./training/Dockerfile -t training_image .
docker run --name training_container training_image

Explanation:
	•	docker build -f ./training/Dockerfile -t training_image .: Builds a Docker image named training_image using the Dockerfile located in the training directory.
	•	docker run --name training_container training_image: Runs a container named training_container from the training_image, initiating the training process inside the Docker environment.

#### 3.Copy Trained Models and MLflow Runs from the Container

After training completes, transfer the trained models and MLflow tracking data from the Docker container to your local machine. This step ensures that you have access to the model artifacts and experiment tracking information outside the containerized environment.

docker cp training_container:/app/models ./models
docker cp training_container:/app/mlruns ./mlruns

**Note:**  Replace training_container with the actual name or ID of your running Docker container, which can be found using docker ps.

#### 4.Build and Run the Inference Docker Image

Use Docker to build an isolated environment for running inference.

docker build -f ./inference/Dockerfile -t inference_image .

**_Explanation:_**
- docker build -f ./inference/Dockerfile -t inference_image .: Builds a Docker image named inference_image using the Dockerfile located in the inference directory.

docker run --name inference_container inference_image
docker cp inference_container:/app/results ./results

- **Note:** Replace inference_container with the actual name or ID of your running Docker container. You can find the container ID using docker ps.

### MLflow Integration

MLflow is integrated into the project to facilitate experiment tracking and model management, enhancing reproducibility and organization of machine learning workflows.

Execute:
mlflow ui

Access the UI by navigating to http://localhost:5000 in your web browser.

## Project structure:

This project has a modular structure, where each folder has a specific duty.

ML_basic_example
├── data                      # Data files used for training and inference
│   ├── iris_train.csv
│   └── iris_inference.csv
├── data_process              # Scripts for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Directory to store trained models
│   └── 26.01.2025_18.51.pth
├── training                  # Scripts and Dockerfiles for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── results                   # Directory to store inference results
│   └── 26.01.2025_19.00.csv
├── utils.py                  # Utility functions and classes
├── settings.json             # Configuration settings
├── requirements.txt          # Project dependencies
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation


## Settings

All configurable parameters and settings for the project are managed through the settings.json file. This centralized configuration allows for easy adjustments without the need to modify the codebase directly.

### **settings.json Structure:**

{
    "general": {
        "random_state": 42,
        "status": "test",
        "datetime_format": "%d.%m.%Y_%H.%M",
        "data_dir": "data",
        "models_dir": "models",
        "results_dir": "results"
    },
    "train": {
        "table_name": "iris_train.csv",
        "data_sample": 105,
        "test_size": 0.3,
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 10
    },
    "inference": {
        "inp_table_name": "iris_inference.csv",
        "model_name": "prod_model.pth",
        "use_latest_model": true
    }
}



