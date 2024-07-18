#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Forward declarations
unsigned char* conv2dWithCuda(unsigned char* img, const int* kernel, size_t krows, size_t kcols, size_t irows, size_t icols, size_t channels);

int main() {
    //To Do: take cmd arguments
    const char* inputImagePath = "C:/Users/shubs/OneDrive/Documents/fullHD.jpg"; 
    const char* outputImagePath = "output.jpg";

    // Load image using OpenCV
    cv::Mat img = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return 1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    size_t imgSize = width * height * channels * sizeof(unsigned char);

    // Define the vertical edge detection kernel
    const int cudaKernel[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    };

    cv::Mat openCVkernel = (cv::Mat_<float>(3, 3) <<
        1, 0, -1,
        2, 0, -2,
        1, 0, -1);

    // Perform 2D convolution using OpenCV
    cv::Mat outputImg1;
    auto start = std::chrono::high_resolution_clock::now();
    cv::filter2D(img, outputImg1, -1, openCVkernel);
    cv::imwrite("outputOpCV1.jpg", outputImg1);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken by OpenCV: " << duration.count() << " ms" << std::endl;

    // Perform 2D convolution using CUDA
    start = std::chrono::high_resolution_clock::now();
    unsigned char* result = conv2dWithCuda(img.data, (const int*)cudaKernel, 3, 3, height, width, channels);
    end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    duration = end - start;
    std::cout << "Time taken by CUDA: " << duration.count() << " ms" << std::endl;

    if (result) {
        // Create an OpenCV Mat from the result
        cv::Mat outputImg(height, width, CV_8UC1, result);

        // Save the result image
        cv::imwrite(outputImagePath, outputImg);

        // Free the allocated memory
        free(result);
    }

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed!" << std::endl;
        return 1;
    }

    return 0;
}
