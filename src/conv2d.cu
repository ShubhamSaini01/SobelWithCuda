#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

// CUDA kernel for 2D convolution
__global__ void conv2dKernel(unsigned char* dst, const unsigned char* img, const int* kernel, size_t krows, size_t kcols, size_t irows, size_t icols, size_t channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < irows && col < icols) {
        int dx = kcols / 2;
        int dy = krows / 2;
        for (size_t c = 0; c < channels; c++) {
            float tmp = 0.0f;
            for (int k = 0; k < krows; ++k) {
                for (int l = 0; l < kcols; ++l) {
                    int x = col - dx + l;
                    int y = row - dy + k;
                    if (x >= 0 && x < icols && y >= 0 && y < irows) {
                        tmp += img[(y * icols + x) * channels + c] * kernel[k * kcols + l];
                    }
                }
            }
            dst[(row * icols + col) * channels + c] = fminf(fmaxf(tmp, 0.0f), 255.0f);
        }
    }
}

// Function to perform 2D convolution using CUDA
unsigned char* conv2dWithCuda(unsigned char* img, const int* kernel, size_t krows, size_t kcols, size_t irows, size_t icols, size_t channels) {
    unsigned char* dev_img = nullptr;
    int* dev_kernel = nullptr;
    unsigned char* dev_dst = nullptr;
    unsigned char* result = nullptr;
    cudaError_t cudaStatus;
    size_t imgSize = irows * icols * channels * sizeof(unsigned char);
    size_t kernelSize = krows * kcols * sizeof(int);

    // Allocate host memory for the result
    result = (unsigned char*)std::malloc(imgSize);
    if (!result) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        goto Error;
    }

    // Choose which GPU to run on
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        goto Error;
    }

    // Allocate GPU buffers for the image, kernel, and output
    cudaStatus = cudaMalloc((void**)&dev_img, imgSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dev_img!" << std::endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_kernel, kernelSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dev_kernel!" << std::endl;
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_dst, imgSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for dev_dst!" << std::endl;
        goto Error;
    }

    // Copy image and kernel from host to device
    cudaStatus = cudaMemcpy(dev_img, img, imgSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for dev_img!" << std::endl;
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_kernel, kernel, kernelSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for dev_kernel!" << std::endl;
        goto Error;
    }

    // Launch the convolution kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((icols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (irows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    conv2dKernel << <numBlocks, threadsPerBlock >> > (dev_dst, dev_img, dev_kernel, krows, kcols, irows, icols, channels);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "conv2dKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching conv2dKernel!" << std::endl;
        goto Error;
    }

    // Copy the result from device to host
    cudaStatus = cudaMemcpy(result, dev_dst, imgSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for result!" << std::endl;
        goto Error;
    }

Error:
    cudaFree(dev_img);
    cudaFree(dev_kernel);
    cudaFree(dev_dst);

    if (cudaStatus != cudaSuccess) {
        std::free(result);
        result = nullptr;
    }

    return result;
}
