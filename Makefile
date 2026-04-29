# Define the compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++11 -Iinclude
LDFLAGS = 

# Define directories
SRC_DIR = src
BIN_DIR = bin

# Define target
TARGET = $(BIN_DIR)/edgeDetectionPipeline
SRC = $(SRC_DIR)/edgeDetectionPipeline.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)/*
