#!/bin/bash

# Define backend and frontend directories
BACKEND_DIR="./app"
FRONTEND_DIR="../"
CONDA_ENV_NAME="f5-tts" # Change this to your environment name

# Function to activate Conda environment
activate_conda_env() {
  echo "Activating Conda environment: $CONDA_ENV_NAME"
  source /opt/anaconda3/etc/profile.d/conda.sh  # Adjust this path based on your Conda installation
  conda activate $CONDA_ENV_NAME
}

# Function to start backend (FastAPI)
start_backend() {
  echo "Starting FastAPI backend..."
  cd $BACKEND_DIR
  nohup uvicorn main:app --reload &
  BACKEND_PID=$!
  echo "Backend running with PID $BACKEND_PID"
}

# Function to start frontend (HTML)
start_frontend() {
  echo "Starting HTML frontend on port 3000..."
  cd $FRONTEND_DIR
  python3 -m http.server 3000 --directory . --bind 127.0.0.1 &
  FRONTEND_PID=$!
  echo "Frontend running with PID $FRONTEND_PID"
}

# Function to stop both servers
stop_servers() {
  echo "Stopping servers..."
  if [ ! -z "$BACKEND_PID" ]; then
    kill $BACKEND_PID
    echo "Backend stopped."
  fi

  if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID
    echo "Frontend stopped."
  fi
}

# Activate Conda environment
activate_conda_env

# Start servers
start_backend
start_frontend

# Catch exit and stop servers
trap stop_servers EXIT

# Wait for both processes
wait