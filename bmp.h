#pragma once
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>

struct BitMapFileHeader {
	unsigned short Filetype;
	unsigned int FileSize;
	unsigned short Reserved1;
	unsigned short Reserved2;
	unsigned int OffsetData;
};

struct BitMapInfoHeader {
	unsigned int BSize;
	int Width;
	int Height;
	unsigned short Planes;
	unsigned short BitCount;
	unsigned int Compression;
	unsigned int SizeImage;
	int XPixelsPerMeter;
	int YPixelsPerMeter;
	unsigned int ColorsUsed;
	unsigned int ColorsImportant;
};

class BMP
{
public:

	unsigned char** reds;
	unsigned char** greens;
	unsigned char** blues;
	int rows;
	int cols;

	std::vector<char> buffer;
	std::vector<char> ColorData;

	BitMapFileHeader FileHeader;
	BitMapInfoHeader InfoHeader;

	BMP();
	void LoadFromFile(char*);
	void OutputFileData();
	int WriteImageToFile(char*);
};
