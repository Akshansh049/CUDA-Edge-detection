#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <math.h>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <cuda_runtime.h>

// Helper to check for CUDA errors
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// ---------------------------------------------------------
// CUDA KERNELS
// ---------------------------------------------------------

// Kernel 1: Convert RGB to Grayscale
__global__ void rgbToGrayscale(unsigned char *d_in, unsigned char *d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx_in = (y * width + x) * 3; // 3 channels
        int idx_out = y * width + x;      // 1 channel

        unsigned char r = d_in[idx_in];
        unsigned char g = d_in[idx_in + 1];
        unsigned char b = d_in[idx_in + 2];

        // Standard luminosity formula
        d_out[idx_out] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Kernel 2: Apply a 5x5 Gaussian Blur to reduce noise
__global__ void gaussianBlur(unsigned char *d_in, unsigned char *d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 5x5 Gaussian Kernel approx
    float blurKernel[5][5] = {
        {1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f},
        {4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f},
        {6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f},
        {4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f},
        {1/256.0f,  4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f}
    };

    if (x < width && y < height) {
        float sum = 0.0f;
        int half = 2;

        for (int i = -half; i <= half; i++) {
            for (int j = -half; j <= half; j++) {
                int px = min(max(x + j, 0), width - 1);
                int py = min(max(y + i, 0), height - 1);
                
                sum += d_in[py * width + px] * blurKernel[i + half][j + half];
            }
        }
        d_out[y * width + x] = (unsigned char)min(max((int)sum, 0), 255);
    }
}

// Kernel 3: Sobel Edge Detection
__global__ void sobelEdgeDetection(unsigned char *d_in, unsigned char *d_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    if (x < width && y < height) {
        float sumX = 0;
        float sumY = 0;
        
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int px = min(max(x + j, 0), width - 1);
                int py = min(max(y + i, 0), height - 1);
                
                int pixel = d_in[py * width + px];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }
        
        float magnitude = sqrt(sumX * sumX + sumY * sumY);
        d_out[y * width + x] = (unsigned char)min(max((int)magnitude, 0), 255);
    }
}

// ---------------------------------------------------------
// CPU HOST PIPELINE
// ---------------------------------------------------------

void processImagePipeline(const std::string& input_path, const std::string& output_path_base) {
    int width, height, channels;
    // Load image as 3 channels (RGB)
    unsigned char* h_src = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
    
    if (!h_src) {
        std::cerr << "Failed to load image: " << input_path << std::endl;
        return;
    }

    size_t img_size_rgb = width * height * 3 * sizeof(unsigned char);
    size_t img_size_gray = width * height * sizeof(unsigned char);

    // Allocate memory on GPU
    unsigned char *d_rgb, *d_gray, *d_blurred, *d_edges;
    checkCudaErrors(cudaMalloc((void**)&d_rgb, img_size_rgb));
    checkCudaErrors(cudaMalloc((void**)&d_gray, img_size_gray));
    checkCudaErrors(cudaMalloc((void**)&d_blurred, img_size_gray));
    checkCudaErrors(cudaMalloc((void**)&d_edges, img_size_gray));

    // Copy RGB image to GPU
    checkCudaErrors(cudaMemcpy(d_rgb, h_src, img_size_rgb, cudaMemcpyHostToDevice));

    // Setup Block and Grid dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Setup CUDA Events for detailed profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // --- Launch Grayscale Kernel ---
    cudaEventRecord(start);
    rgbToGrayscale<<<numBlocks, threadsPerBlock>>>(d_rgb, d_gray, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "    [-] Grayscale Kernel Time: " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;

    // --- Launch Gaussian Blur Kernel ---
    cudaEventRecord(start);
    gaussianBlur<<<numBlocks, threadsPerBlock>>>(d_gray, d_blurred, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "    [-] Gaussian Blur Time:    " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;

    // --- Launch Sobel Edge Detection Kernel ---
    cudaEventRecord(start);
    sobelEdgeDetection<<<numBlocks, threadsPerBlock>>>(d_blurred, d_edges, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "    [-] Sobel Edge Time:       " << std::fixed << std::setprecision(3) << milliseconds << " ms" << std::endl;

    // Allocate memory on host for the results
    unsigned char* h_gray = new unsigned char[width * height];
    unsigned char* h_blurred = new unsigned char[width * height];
    unsigned char* h_edges = new unsigned char[width * height];

    // Copy the results back to CPU
    checkCudaErrors(cudaMemcpy(h_gray, d_gray, img_size_gray, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_blurred, d_blurred, img_size_gray, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_edges, d_edges, img_size_gray, cudaMemcpyDeviceToHost));

    // Determine output file paths
    std::string base_name = output_path_base;
    size_t dot_pos = base_name.find_last_of('.');
    if (dot_pos != std::string::npos) {
        base_name = base_name.substr(0, dot_pos);
    }

    std::string out_gray = base_name + "_1_gray.png";
    std::string out_blur = base_name + "_2_blurred.png";
    std::string out_edge = base_name + "_3_edges.png";

    // Save all intermediate results
    stbi_write_png(out_gray.c_str(), width, height, 1, h_gray, width);
    std::cout << "    [+] Saved: " << out_gray << std::endl;

    stbi_write_png(out_blur.c_str(), width, height, 1, h_blurred, width);
    std::cout << "    [+] Saved: " << out_blur << std::endl;

    stbi_write_png(out_edge.c_str(), width, height, 1, h_edges, width);
    std::cout << "    [+] Saved: " << out_edge << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blurred);
    cudaFree(d_edges);
    delete[] h_gray;
    delete[] h_blurred;
    delete[] h_edges;
    stbi_image_free(h_src);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>" << std::endl;
        return 1;
    }

    std::string input_dir = argv[1];
    std::string output_dir = argv[2];

    DIR* dir;
    struct dirent* ent;
    
    // Open input directory
    if ((dir = opendir(input_dir.c_str())) != NULL) {
        std::cout << "========================================" << std::endl;
        std::cout << " Starting Batch Edge Detection Pipeline" << std::endl;
        std::cout << "========================================" << std::endl;

        while ((ent = readdir(dir)) != NULL) {
            std::string filename(ent->d_name);
            // Process common image extensions
            if (filename.length() > 4 && 
                (filename.substr(filename.length() - 4) == ".png" || 
                 filename.substr(filename.length() - 4) == ".jpg" ||
                 filename.substr(filename.length() - 5) == ".tiff" ||
                 filename.substr(filename.length() - 5) == ".jpeg")) {
                
                std::string input_path = input_dir + "/" + filename;
                std::string output_path = output_dir + "/" + filename;
                
                std::cout << "\n[*] Processing: " << filename << std::endl;
                processImagePipeline(input_path, output_path);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open input directory: " << input_dir << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << " Batch Processing Complete. Outputs saved." << std::endl;
    std::cout << "========================================" << std::endl;
    return EXIT_SUCCESS;
}
