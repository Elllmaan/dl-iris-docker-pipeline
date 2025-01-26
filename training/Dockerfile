# Use a lightweight Python base image
FROM python:3.10-slim

# Define build argument for settings file
ARG settings_name=settings.json

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY data/ /app/data/
COPY training/ /app/training/
COPY utils.py /app/
COPY ${settings_name} /app/settings.json

# Set environment variables
ENV CONF_PATH=/app/settings.json

# Define the entrypoint command for running training
CMD ["python", "training/train.py"]