
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

    BMP(int32_t width, int32_t height, bool has_alpha = true) {
    	// ...
    }

    void write(const char *fname) {
    	// ...
    }

private:
    // ...
};