
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "BMP.h"

enum Color{Red, Blue, Green};

BMP persistColor(BMP *bmp, Color color);

__global__ void PersistColor(BMP *bmp, Color color)
{
	// The goal of this kernel is to remove all colors except for black, white, and the designated color (red, blue, or green)
}

int main()
{
	BMP bmp;
	bmp.LoadFromFile("cat.bmp");
	bmp.OutputFileData();

	BMP outputBMP = persistColor(&bmp, Blue);

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "cudaDeviceReset failed!");
	    return 1;
	}


	printf("\nProgram completed.\n");
	std::getchar();
	return 0;
  
}

BMP persistColor(BMP *bmp, Color color)
{
	//TODO
	// 1. Add Timing
	// 
	BMP *dev_bmp;
	BMP *host_output_bmp;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMallocHost((void**)&host_output_bmp, sizeof(BMP));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_bmp, sizeof(BMP));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_bmp, bmp, sizeof(BMP), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	PersistColor<<<1, sizeof(BMP) >>>(dev_bmp, color);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "PersistColor launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(host_output_bmp, dev_bmp, sizeof(BMP), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_bmp);
	return *host_output_bmp;
}

