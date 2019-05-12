
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <gdiplus.h>
using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")
#include <stdio.h>
#include <iostream>
#include <chrono>

#define DEBUG			0

#define BLUE_BITMASK	0xff
#define GREEN_BITMASK	0xff00
#define RED_BITMASK		0xff0000
#define ALPHA_BITMASK	0xff000000	// Bitmasks are used to extract individual color values from the pixel information

#define BLUE_SHIFT		0
#define GREEN_SHIFT		8
#define RED_SHIFT		16
#define ALPHA_SHIFT		24			// Number of bytes the color values are shifted in the pixel information.

#define BLUE_GRAYSCALE	0.11
#define GREEN_GRAYSCALE	0.59
#define RED_GRAYSCALE	0.3

#define BLUE_THRESHOLD	140			//These thresholds adjust how "vibrant" a color has to be to persist.  
#define GREEN_THRESHOLD	125
#define RED_THRESHOLD	100

#define NUM_THREADS		1024

#define	FILE_DIRECTORY		L""  // File directory that bitmap images are in.  This is only used if the DEBUG flag is set, otherwise it is determined by command line arguments.
#define HOST_SUFFIX			L"_HOST.bmp"
#define DEVICE_SUFFIX		L"_DEVICE.bmp"

const wchar_t *BMP_ENCODER_CLSID = L"{557cf400-1a04-11d3-9a73-0000f81ef32e}";

wchar_t * gFileDirectory;

enum PersistColor { Red, Blue, Green };

void SaveBitmap(Bitmap *processedBitmap, wchar_t* fileName, wchar_t* directoryName, wchar_t* fileSuffix);
__host__ UINT* PersistColor_HOST(UINT *pixels, PersistColor color, unsigned int size);
UINT* persistColor_DEVICE(UINT *pixels, PersistColor color, unsigned int size);
void RunTest(wchar_t * InputFileName, wchar_t * OutputFileName, PersistColor pColor);

__global__ void PersistColor_GPU(UINT *pixels, UINT *outPixels, PersistColor color)
{
	// The goal of this kernel is to remove all colors except for black, white, and the designated color (red, blue, or green)
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	UINT pixel = pixels[thread_idx];
	int b = pixel & BLUE_BITMASK;
	int g = (pixel & GREEN_BITMASK) >> GREEN_SHIFT;
	int r = (pixel & RED_BITMASK) >> RED_SHIFT;
	int a = (pixel & ALPHA_BITMASK) >> ALPHA_SHIFT;

	int grayScale = (int)((r * RED_GRAYSCALE) + (g * GREEN_GRAYSCALE) + (b * BLUE_GRAYSCALE));

	switch (color)
	{
		case Red:
			if (r > RED_THRESHOLD && g < GREEN_THRESHOLD && b < BLUE_THRESHOLD)
			{
				r = r;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			else
			{
				r = grayScale;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			break;
		case Green:
			if (r < RED_THRESHOLD && g > GREEN_THRESHOLD && b < BLUE_THRESHOLD)
			{
				r = grayScale;
				b = grayScale;
				g = g;
				a = a;
			}
			else
			{
				r = grayScale;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			break;
		case Blue:
			if (r < RED_THRESHOLD && g < GREEN_THRESHOLD && b > BLUE_THRESHOLD)
			{
				r = grayScale;
				b = b;
				g = grayScale;
				a = a;
			}
			else
			{
				r = grayScale;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			break;
		default:
			break;
	}

	UINT newpixel = (a << ALPHA_SHIFT);
	newpixel = newpixel | (r << RED_SHIFT);
	newpixel = newpixel | (g << GREEN_SHIFT);
	newpixel = newpixel | (b);
	outPixels[thread_idx] = newpixel;
}

__host__ UINT* PersistColor_HOST(UINT *pixels, PersistColor color, unsigned int size)
{
	UINT *processedPixels;
	processedPixels = new UINT[size];
	for (size_t i = 0; i < size; i++)
	{
		unsigned int pixel = pixels[i];
		int b = pixel & BLUE_BITMASK;
		int g = (pixel & GREEN_BITMASK) >> GREEN_SHIFT;
		int r = (pixel & RED_BITMASK) >> RED_SHIFT;
		int a = (pixel & ALPHA_BITMASK) >> ALPHA_SHIFT;

		int grayScale = (int)((r * RED_GRAYSCALE) + (g * GREEN_GRAYSCALE) + (b * BLUE_GRAYSCALE));

		switch (color)
		{
		case Red:
			if (r > RED_THRESHOLD && g < GREEN_THRESHOLD && b < BLUE_THRESHOLD)
			{
				r = r;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			else
			{
				r = grayScale;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			break;
		case Green:
			if (r < RED_THRESHOLD && g > GREEN_THRESHOLD && b < BLUE_THRESHOLD)
			{
				r = grayScale;
				b = grayScale;
				g = g;
				a = a;
			}
			else
			{
				r = grayScale;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			break;
		case Blue:
			if (r < RED_THRESHOLD && g < GREEN_THRESHOLD && b > BLUE_THRESHOLD)
			{
				r = grayScale;
				b = b;
				g = grayScale;
				a = a;
			}
			else
			{
				r = grayScale;
				b = grayScale;
				g = grayScale;
				a = a;
			}
			break;
		default:
			break;
		}

		UINT newpixel = (a << ALPHA_SHIFT);
		newpixel = newpixel | (r << RED_SHIFT);
		newpixel = newpixel | (g << GREEN_SHIFT);
		newpixel = newpixel | (b);
		processedPixels[i] = newpixel;
	}
	return processedPixels;
}

UINT* persistColor_DEVICE(UINT *pixels, PersistColor color, unsigned int size)
{
	const unsigned int num_elements = size;
	const unsigned int num_threads = NUM_THREADS;
	const unsigned int num_blocks = num_elements / num_threads;
	const unsigned int num_bytes = num_elements * sizeof(UINT);
	UINT *dev_pixels_in;
	UINT *dev_pixels_out;
	UINT *host_pixels_input;
	UINT *host_pixels_output;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMallocHost((void**)&host_pixels_output, sizeof(UINT) * num_elements);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed!");
		goto Error;
	}
	cudaStatus = cudaMallocHost((void**)&host_pixels_input, sizeof(UINT) * num_elements);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocHost failed!");
		goto Error;
	}

	for (size_t i = 0; i < size; i++)
	{
		host_pixels_input[i] = pixels[i];
	}

	cudaStatus = cudaMalloc((void**)&dev_pixels_in, sizeof(UINT) * num_elements);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_pixels_out, sizeof(UINT) * num_elements);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_pixels_in, host_pixels_input, sizeof(UINT) * num_elements, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	PersistColor_GPU <<<num_blocks, num_threads >> >(dev_pixels_in, dev_pixels_out, color);

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

	cudaStatus = cudaMemcpy(host_pixels_output, dev_pixels_out, sizeof(UINT) * num_elements, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaFree(dev_pixels_in);
	cudaFree(dev_pixels_out);
	return host_pixels_output;
Error:
	fprintf(stderr, "ERROR!");
	std::getchar();
	cudaFree(dev_pixels_in);
	cudaFree(dev_pixels_out);
	return host_pixels_output;
}

void RunTest(wchar_t * InputFileName, wchar_t * OutputFileName, PersistColor pColor)
{
	UINT *pixels;
	UINT *processedPixels_HOST;
	UINT *processedPixels_DEVICE;
	int nWidth;
	int nHeight;
	int nStride1;
	wchar_t *directory;
	if (DEBUG)
	{
		directory = FILE_DIRECTORY;
	}
	else
	{
		directory = gFileDirectory;
	}


	int fileNameLength = wcslen(directory) + wcslen(InputFileName);

	wchar_t* InputFileNameComplete = new wchar_t[fileNameLength + 1];
	wcscpy(InputFileNameComplete, directory);
	wcscat(InputFileNameComplete, InputFileName);

	GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	{
		Bitmap image(InputFileNameComplete, true);
		nWidth = image.GetWidth();
		nHeight = image.GetHeight();

		std::cout << "Image size: " << nWidth << " x " << nHeight << std::endl;
		std::cout << std::endl;

		Rect rect1(0, 0, nWidth, nHeight);

		pixels = new UINT[nWidth * nHeight];
		processedPixels_HOST = new UINT[nWidth * nHeight];

		memset(pixels, 0, sizeof(UINT)*nWidth*nHeight);
		memset(processedPixels_HOST, 0, sizeof(UINT)*nWidth*nHeight);

		BitmapData bitmapData;
		memset(&bitmapData, 0, sizeof(bitmapData));
		image.LockBits(
			&rect1,
			ImageLockModeRead,
			PixelFormat32bppARGB,
			&bitmapData);

		nStride1 = bitmapData.Stride;
		if (nStride1 < 0)
			nStride1 = -nStride1;

		UINT* DestPixels = (UINT*)bitmapData.Scan0;

		for (UINT row = 0; row < bitmapData.Height; ++row)
		{
			for (UINT col = 0; col < bitmapData.Width; ++col)
			{
				pixels[row * nWidth + col] = DestPixels[row * nStride1 / 4 + col];
			}
		}

		image.UnlockBits(&bitmapData);
	}

	unsigned int size = nWidth * nHeight;

	auto now = std::chrono::high_resolution_clock::now();

	auto deviceStart = std::chrono::high_resolution_clock::now();
	processedPixels_DEVICE = persistColor_DEVICE(pixels, pColor, size);
	auto deviceStop= std::chrono::high_resolution_clock::now();
	auto hostStart = std::chrono::high_resolution_clock::now();
	processedPixels_HOST = PersistColor_HOST(pixels, pColor, size);
	auto hostStop = std::chrono::high_resolution_clock::now();

	auto deviceDuration = std::chrono::duration_cast<std::chrono::microseconds>(deviceStop - deviceStart);
	auto hostDuration = std::chrono::duration_cast<std::chrono::microseconds>(hostStop - hostStart);

	std::cout << "Device processing took " << deviceDuration.count() << " microseconds" << std::endl;
	std::cout << "Host processing took " << hostDuration.count() << " microseconds" << std::endl;

	std::cout << "____________________________________________________" << std::endl;

	Bitmap *processedBitmap_HOST = new Bitmap(nWidth, nHeight, nStride1, (PixelFormat)PixelFormat32bppARGB, (BYTE*)processedPixels_HOST);
	Bitmap *processedBitmap_DEVICE = new Bitmap(nWidth, nHeight, nStride1, (PixelFormat)PixelFormat32bppARGB, (BYTE*)processedPixels_DEVICE);

	wchar_t *outputFile = OutputFileName;
	wchar_t *hostSuffix = HOST_SUFFIX;
	wchar_t *deviceSuffix = DEVICE_SUFFIX;
	
	SaveBitmap(processedBitmap_HOST, outputFile, directory, hostSuffix);
	SaveBitmap(processedBitmap_DEVICE, outputFile, directory, deviceSuffix);

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	//std::getchar();
	GdiplusShutdown(gdiplusToken);
}

void SaveBitmap(Bitmap *processedBitmap, wchar_t* fileName, wchar_t* directoryName, wchar_t* fileSuffix)
{
	CLSID bmpClsid;
	CLSIDFromString(BMP_ENCODER_CLSID, &bmpClsid);

	int fileNameLength = wcslen(directoryName) + wcslen(fileName) + wcslen(fileSuffix);

	wchar_t* completeFileName = new wchar_t[fileNameLength + 1];
	wcscpy(completeFileName, directoryName);
	wcscat(completeFileName, fileName);
	wcscat(completeFileName, fileSuffix);

	processedBitmap->Save(completeFileName, &bmpClsid, NULL);
}



int main(int argc, char* argv[])
{
	if (!DEBUG)
	{
		if (argc != 2)
		{
			std::cout << "ERROR: Expected directory parameter."<< std::endl;
			std::cout << "Usage: program.exe <BitMapDirectory>" << std::endl;
			std::getchar();
			return 1;
		}
		size_t directoryLength = strlen(argv[1]);
		gFileDirectory = new wchar_t[directoryLength + 1];
		mbstowcs(gFileDirectory, argv[1], directoryLength + 1);
		std::cout << "File Directory set to " << argv[1] << std::endl;
		std::cout << std::endl;
	}
	std::cout << "## Starting Blue Test..." << std::endl;
	RunTest(L"blue_bus.bmp", L"BLUE", Blue);
	std::cout << std::endl;
	std::cout << "## Starting Green Test..." << std::endl;
	RunTest(L"green_bus.bmp", L"GREEN", Green);
	std::cout << std::endl;
	std::cout << "## Starting Red Test..." << std::endl;
	RunTest(L"red_bus.bmp", L"RED", Red);
	std::cout << "## Starting Blue Test..." << std::endl;
	RunTest(L"blue_bus.bmp", L"BLUE", Blue);
	std::cout << std::endl;

	//std::cout << "## Starting Large Test..." << std::endl;
	//RunTest(L"largebitmap.bmp", L"LARGE", Green);

	std::getchar();
	return 0;
  
}


