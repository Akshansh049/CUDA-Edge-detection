# CUDA at Scale for the Enterprise - Capstone Project

## Overview
This project fulfills the Capstone assignment by implementing a custom **Batch Image Edge Detection Pipeline** on the GPU. Instead of simply wrapping pre-built libraries, this project highlights direct hardware manipulation by utilizing **Custom CUDA Kernels** to process images from the USC Viterbi SIPI Image Database.

The pipeline executes the following stages entirely on the GPU:
1. **RGB to Grayscale Conversion**: Converts a 3-channel image to 1-channel.
2. **Gaussian Blur (5x5)**: Reduces high-frequency noise and details to improve edge detection.
3. **Sobel Edge Detection**: Applies Sobel horizontal and vertical convolution filters to compute gradient magnitude, highlighting the edges.

This sequence is an excellent demonstration of basic Computer Vision on the GPU, achieving high parallelism without external library dependencies like NPP or OpenCV. 

## Project Structure

```text
CUDA_Project/
├── bin/                       # Compiled executables
├── data/                  
│   ├── input/                 # Original dataset (misc volume)
│   └── output/                # Processed edge-detected images
├── include/               
│   ├── stb_image.h            # Lightweight image reading library
│   └── stb_image_write.h      # Lightweight image writing library
├── src/
│   └── edgeDetectionPipeline.cu # Main CUDA source code
├── Makefile                   # Build script
└── run.sh                     # Pipeline execution script
```

## Why `stb_image`?
The standard lightweight image I/O headers (`stb_image.h` and `stb_image_write.h`) were chosen to read and write images without requiring complex external library installations (like FreeImage, libtiff, or OpenCV) which can be notoriously complex to set up in the Coursera Lab environment. 

## Execution in Coursera Lab

To run the full pipeline (which will automatically download the SIPI Misc data, compile the CUDA program, and process the images):

```bash
# Ensure the run script is executable
chmod +x run.sh

# Execute the pipeline
./run.sh
```

Alternatively, you can run the steps manually:
```bash
# 1. Download Dataset (or place images manually in data/input)
wget https://sipi.usc.edu/database/misc.zip
unzip misc.zip -d data/
mv data/misc/* data/input/

# 2. Build project
make

# 3. Execute batch processing
./bin/edgeDetectionPipeline data/input data/output
```

## Evidence & Demonstration

After running the pipeline, you will find the original images in `data/input` and their edge-detected counterparts in `data/output`. You can provide screenshots of these outputs as proof of execution for peer grading. The included `run.sh` terminal output serves as log file evidence.
