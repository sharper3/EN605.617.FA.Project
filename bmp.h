
#include <fstream>
#include <vector>
#include <stdexcept>

struct BMPFileHeader {
    uint16_t Filetype;          
    uint32_t FileSize;          
    uint16_t Reserved1;          
    uint16_t Reserved2;          
    uint32_t OffsetData;        
};

struct BMPInfoHeader {
    uint32_t BSize;                      
    int32_t Width;                      
    int32_t Height;                     
    uint16_t Planes;                    
    uint16_t BitCount;                 
    uint32_t Compression;               
    uint32_t SizeImage;                
    int32_t XPixelsPerMeter;
    int32_t YPixelsPerMeter;
    uint32_t ColorsUsed;               
    uint32_t ColorsImportant;   
};

struct BMPColorHeader {
    uint32_t RedMask;         // Bit mask for the red channel
    uint32_t GreenMask;       // Bit mask for the green channel
    uint32_t BlueMask;        // Bit mask for the blue channel
    uint32_t AlphaMask;       // Bit mask for the alpha channel
    uint32_t ColorSpaceType; // Default "sRGB" (0x73524742)
    uint32_t unused[16];                // Unused data for sRGB color space
};

struct BMP {
    BMPFileHeader FileHeader;
    BMPInfoHeader BMPInfoHeader;
    BMPColorHeader BMPColorHeader;
    std::vector<char> Data;

    BMP(const char *fname) {
        std::cout << "reading file";
        read(fname);
    }

    void read(const char *fname) {
    	std::ifstream inp(fname, std::ios_base::binary);
        inp.read((char*)&FileHeader, sizeof(FileHeader));
        inp.read((char*)&BMPInfoHeader, sizeof(BMPInfoHeader));
        inp.read((char*)&BMPColorHeader, sizeof(BMPColorHeader));
    }

    void write(const char *fname) {
    	// ...
    }

    void outPutBMPInformation()
    {
        outPutFileHeaderInfo();
        outPutInfoHeaderInfo();
        outPutColorHeaderInfo();
    }

    void outPutFileHeaderInfo()
    {
        std::cout << "File Header"<< std::endl;
        std::cout << "_____________________________________"<< std::endl;
        std::cout << "Filetype: " << FileHeader.Filetype << std::endl;
        std::cout << "FileSize: " << FileHeader.FileSize << std::endl;
        std::cout << "Reserved1: " << FileHeader.Reserved1 << std::endl;
        std::cout << "Reserved2: " << FileHeader.Reserved2 << std::endl;
        std::cout << "OffsetData: " << FileHeader.OffsetData << std::endl;
        std::cout << std::endl;
    }

    void outPutInfoHeaderInfo()
    {
        std::cout << "Info Header"<< std::endl;
        std::cout << "_____________________________________"<< std::endl;
        std::cout << "BSize: " << BMPInfoHeader.BSize << std::endl;
        std::cout << "Width: " << BMPInfoHeader.Width << std::endl;
        std::cout << "Height: " << BMPInfoHeader.Height << std::endl;
        std::cout << "Planes: " << BMPInfoHeader.Planes << std::endl;
        std::cout << "BitCount: " << BMPInfoHeader.BitCount << std::endl;
        std::cout << "Compression: " << BMPInfoHeader.Compression << std::endl;
        std::cout << "SizeImage: " << BMPInfoHeader.SizeImage << std::endl;
        std::cout << "XPixelsPerMeter: " << BMPInfoHeader.XPixelsPerMeter << std::endl;
        std::cout << "YPixelsPerMeter: " << BMPInfoHeader.YPixelsPerMeter << std::endl;
        std::cout << "ColorsUsed: " << BMPInfoHeader.ColorsUsed << std::endl;
        std::cout << "ColorsImportant: " << BMPInfoHeader.ColorsImportant << std::endl;
        std::cout << std::endl; 
    }

    void outPutColorHeaderInfo()
    {
        std::cout << "Color Header"<< std::endl;
        std::cout << "_____________________________________"<< std::endl;
        std::cout << "RedMask: " << BMPColorHeader.RedMask << std::endl;
        std::cout << "GreenMask: " << BMPColorHeader.GreenMask << std::endl;
        std::cout << "BlueMask: " << BMPColorHeader.BlueMask << std::endl;
        std::cout << "AlphaMask: " << BMPColorHeader.AlphaMask << std::endl;
        std::cout << "ColorSpaceType: " << BMPColorHeader.ColorSpaceType << std::endl;
        std::cout << std::endl; 

    }

private:
    // ...
};