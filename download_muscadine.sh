#!/bin/bash

# --- Configuration ---
USER="ng739"
HOST="muscadine-node-1.hpc.msstate.edu"
REMOTE_DIR="/work/users/ng739/cubesat/model/cubesat/yolo_model/runs/"
LOCAL_DIR="./results"

# --- Execution ---
echo " Connecting to Muscadine cluster to sync YOLO results..."

# Create the local folder if it doesn't exist yet
mkdir -p "$LOCAL_DIR"

# Run the smart sync
rsync -avz --progress "$USER@$HOST:$REMOTE_DIR" "$LOCAL_DIR"

echo " Sync complete! All graphs and weights are safely in $LOCAL_DIR"
