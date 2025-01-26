# Use a lightweight Python base image
FROM python:3.10-slim
# Define build arguments (optional values)
ARG settings_name=settings.json

# Define the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY data/ /app/data/
COPY models/ /app/models/
COPY inference/ /app/inference/
COPY utils.py /app/
COPY ${settings_name} /app/settings.json

# Set environment variables
ENV CONF_PATH=/app/settings.json

# Define the entrypoint command for running inference
CMD ["python", "inference/run.py"]