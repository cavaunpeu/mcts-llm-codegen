#!/bin/bash

# Check if experiment name is provided
if [ $# -eq 1 ]; then
    # Define the base command with the --build flag to ensure the image is built if necessary
    CMD="docker-compose run --rm app python -m app.experiment --remote --experiment_name $1"
else
    echo "Error: You must specify an experiment name."
    echo "Usage: $0 experiment_name"
    exit 1
fi

# Execute the command
eval $CMD
