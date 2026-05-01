#!/bin/bash
echo "Starting MLflow Server..."
echo "Access the dashboard in your browser at: http://localhost:5000"
mlflow ui --port 5000 --host 0.0.0.0
