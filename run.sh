#!/bin/bash

{
# Setup directories
mkdir -p data/input
mkdir -p data/output

# Download and unzip dataset if it's not present
if [ -z "$(ls -A data/input)" ]; then
    echo "Downloading SIPI dataset (misc)..."
    wget -qO misc.zip https://sipi.usc.edu/database/misc.zip
    unzip -q misc.zip -d data/
    mv data/misc/* data/input/
    rm -rf data/misc misc.zip
    echo "Dataset downloaded and extracted to data/input"
else
    echo "Dataset already present."
fi

# Build
echo "Building project..."
make clean
make

# Execute
if [ -f "bin/edgeDetectionPipeline" ]; then
    echo "Running Edge Detection Pipeline..."
    ./bin/edgeDetectionPipeline data/input data/output
else
    echo "Build failed! Executable not found."
fi

} 2>&1 | tee output.txt
