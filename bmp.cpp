#include "BMP.h"

BMP::BMP()
{
}

void BMP::LoadFromFile(char *filename)
{
	std::ifstream file(filename);
	file.seekg(0, std::ios::end);
	std::streampos length = file.tellg();
	file.seekg(0, std::ios::beg);

	buffer.resize(length);
	file.read(&buffer[0], length);

	int i = 0;

	std::memcpy(&FileHeader.Filetype, &buffer[i], sizeof(FileHeader.Filetype));
	i += sizeof(FileHeader.Filetype);
	std::memcpy(&FileHeader.FileSize, &buffer[i], sizeof(FileHeader.FileSize));
	i += sizeof(FileHeader.FileSize);
	std::memcpy(&FileHeader.Reserved1, &buffer[i], sizeof(FileHeader.Reserved1));
	i += sizeof(FileHeader.Reserved1);
	std::memcpy(&FileHeader.Reserved2, &buffer[i], sizeof(FileHeader.Reserved2));
	i += sizeof(FileHeader.Reserved2);
	std::memcpy(&FileHeader.OffsetData, &buffer[i], sizeof(FileHeader.OffsetData));
	i += sizeof(FileHeader.OffsetData);

	std::memcpy(&InfoHeader.BSize, &buffer[i], sizeof(InfoHeader.BSize));
	i += sizeof(InfoHeader.BSize);
	std::memcpy(&InfoHeader.Width, &buffer[i], sizeof(InfoHeader.Width));
	i += sizeof(InfoHeader.Width);
	std::memcpy(&InfoHeader.Height, &buffer[i], sizeof(InfoHeader.Height));
	i += sizeof(InfoHeader.Height);
	std::memcpy(&InfoHeader.Planes, &buffer[i], sizeof(InfoHeader.Planes));
	i += sizeof(InfoHeader.Planes);
	std::memcpy(&InfoHeader.BitCount, &buffer[i], sizeof(InfoHeader.BitCount));
	i += sizeof(InfoHeader.BitCount);
	std::memcpy(&InfoHeader.Compression, &buffer[i], sizeof(InfoHeader.Compression));
	i += sizeof(InfoHeader.Compression);
	std::memcpy(&InfoHeader.SizeImage, &buffer[i], sizeof(InfoHeader.SizeImage));
	i += sizeof(InfoHeader.SizeImage);
	std::memcpy(&InfoHeader.XPixelsPerMeter, &buffer[i], sizeof(InfoHeader.XPixelsPerMeter));
	i += sizeof(InfoHeader.XPixelsPerMeter);
	std::memcpy(&InfoHeader.YPixelsPerMeter, &buffer[i], sizeof(InfoHeader.YPixelsPerMeter));
	i += sizeof(InfoHeader.YPixelsPerMeter);
	std::memcpy(&InfoHeader.ColorsUsed, &buffer[i], sizeof(InfoHeader.ColorsUsed));
	i += sizeof(InfoHeader.ColorsUsed);
	std::memcpy(&InfoHeader.ColorsImportant, &buffer[i], sizeof(InfoHeader.ColorsImportant));
	i += sizeof(InfoHeader.ColorsImportant);

	ColorData = std::vector<char>(buffer.begin() + i, buffer.end());


	//for (int j = i; j < 100;j++)
	//{
	//	std::cout << std::hex << (int)buffer[j];
	//	//std::cout << buffer[i] << std::endl;
	//}
	//for (int j = 0; j < ColorData.size();j++)
	//{
	//	printf("%x", ColorData[j]);
	//	//std::cout << buffer[i] << std::endl;
	//}

	//for (int y = 0; y < InfoHeader.Height; y++)
	//{
	//	for (int x = 0; x < InfoHeader.Width; x++)
	//	{

	//	}

	//}

}

int BMP::WriteImageToFile(char *filename)
{
	//TODO write image data to file
	return 0;
}

void BMP::OutputFileData()
{

	//printf("File Header:\n");
	//printf("_____________________________________\n");
	//printf("Filetype:  %x\n", FileHeader.Filetype);
	//printf("FileSize:  %d\n", FileHeader.FileSize);
	//printf("Reserved1:  %d\n", FileHeader.Reserved1);
	//printf("Reserved2:  %d\n", FileHeader.Reserved2);
	//printf("OffsetData:  %d\n", FileHeader.OffsetData);
	//printf("\n");

	//printf("Info Header:\n");
	//printf("_____________________________________\n");
	//printf("BSize:  %d\n", InfoHeader.BSize);
	//printf("Width:  %d\n", InfoHeader.Width);
	//printf("Height:  %d\n", InfoHeader.Height);
	//printf("Planes:  %d\n", InfoHeader.Planes);
	//printf("BitCount:  %d\n", InfoHeader.BitCount);
	//printf("Compression:  %d\n", InfoHeader.Compression);
	//printf("SizeImage:  %d\n", InfoHeader.SizeImage);
	//printf("XPixelsPerMeter:  %d\n", InfoHeader.XPixelsPerMeter);
	//printf("YPixelsPerMeter:  %d\n", InfoHeader.YPixelsPerMeter);
	//printf("ColorsUsed:  %d\n", InfoHeader.ColorsUsed);
	//printf("ColorsImportant:  %d\n", InfoHeader.ColorsImportant);
	
	printf("__________________________________________________________________________\n");
	printf("\n");
	printf("IMAGE DATA\n");
	printf("Type:				%d bit\n", InfoHeader.BitCount);
	printf("Dimensions:			%d X %d\n", InfoHeader.Width, InfoHeader.Height);
	printf("Image Size:			%f MB\n", (FileHeader.FileSize/1000000.00));
	printf("Total Pixels:			%d\n", InfoHeader.Width * InfoHeader.Height);
	printf("__________________________________________________________________________\n");
}
