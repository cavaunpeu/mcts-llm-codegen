#!/bin/bash

# Define the base command with the --build flag to ensure the image is built if necessary
CMD="docker-compose run --rm app python -m app.main --remote"

# Append test problem index to the command if provided
if [ $# -eq 1 ]; then
  CMD="$CMD --test_problem_index $1"
fi

# Execute the command
eval $CMD