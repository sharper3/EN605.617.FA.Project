#include <iostream>
#include <fstream>
#include <vector>
#include "bmp.h"



void read_file() {
    std::cout << "starting...";
    std::vector<char> buffer;
    std::ifstream file("cat.bmp");

   
    file.seekg(0,std::ios::end);
    std::streampos length = file.tellg();
    file.seekg(0,std::ios::beg);

    buffer.resize(length);
    file.read(&buffer[0],length);
    std::cout << "reading file...";
    
    for(int i = 0; i < buffer.size();i++)
    {
        std::cout << std::hex << (int)buffer[i];
        //std::cout << buffer[i] << std::endl;
    }


}

int main()
{
    //read_file();
    BMP bmp("cat.bmp");
    std::cout << bmp.BMPInfoHeader.BSize << std::endl;
    return 0;
}