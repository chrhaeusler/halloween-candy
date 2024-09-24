#!/bin/bash

# Check if a virtual environment name was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <new-env-name>"
    exit 1
fi

NEW_ENV_NAME=$1

# Check if requirements.txt exists in the current directory
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found in the current directory."
    exit 1
fi

# Upgrade pip in the current environment
echo
echo "Upgrading pip in the current environment..."
pip install --upgrade pip

# Print the absolute path of the new virtual environment before creation
NEW_ENV_PATH=$(realpath $NEW_ENV_NAME)
echo
echo "Absolute path of the new virtual environment (before setup): $NEW_ENV_PATH"

# Create a new virtual environment
echo
echo "Creating a new virtual environment: $NEW_ENV_NAME"
python -m venv $NEW_ENV_NAME

# Print the absolute path of the new virtual environment after creation
echo
echo "Absolute path of the new virtual environment (after setup): $NEW_ENV_PATH"

# Upgrade pip in the new virtual environment
echo
echo "Upgrading pip in the new virtual environment..."
$NEW_ENV_NAME/bin/pip install --upgrade pip

# Install packages from requirements.txt
echo
echo "Installing packages from requirements.txt..."
$NEW_ENV_NAME/bin/pip install -r requirements.txt

echo
echo "Setup complete. New virtual environment '$NEW_ENV_NAME' is ready."

# Provide instructions for activating the new environment
echo
echo "To activate the new virtual environment, use the following command:"
echo "source $NEW_ENV_NAME/bin/activate"

