#!/bin/bash

# Script to build and run Docker services with optional test problem index

# Build the app service
docker-compose build app

# Set the base command
CMD="docker-compose run --rm app python main.py --remote"

# Append test problem index to the command if provided
if [ $# -eq 1 ]; then
  CMD="$CMD --test_problem_index $1"
fi

# Execute the command
eval $CMD