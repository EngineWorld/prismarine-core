#pragma once

#include "includes.hpp"

namespace Paper {
    const int32_t kW = 0;
    const int32_t kA = 1;
    const int32_t kS = 2;
    const int32_t kD = 3;
    const int32_t kQ = 4;
    const int32_t kE = 5;
    const int32_t kSpc = 6;
    const int32_t kSft = 7;
    const int32_t kC = 8;
    const int32_t kK = 9;

    class PTObject {};
    class Tracer;
    class Intersector;

    const uint32_t _zero = 0;
    const uint32_t _one = 1;

#ifdef USE_FREEIMAGE
    static uint8_t * swapBGRA8(uint8_t * data, size_t size) {
        for (size_t i = 0; i < size;i++) {
            size_t i4 = i * 4;
            uint8_t r = data[i4 + 0];
            data[i4 + 0] = data[i4 + 2];
            data[i4 + 2] = r;
        }
        return data;
    }

    static int loadImage(std::vector<uint8_t> &image, uint32_t& width, uint32_t& height, const std::string &tex) {
        FREE_IMAGE_FORMAT formato = FreeImage_GetFileType(tex.c_str(), 0);
        if (formato == FIF_UNKNOWN) {
            return 0;
        }
        FIBITMAP* imagen = FreeImage_Load(formato, tex.c_str());
        if (!imagen) {
            return 0;
        }
        FIBITMAP* temp = FreeImage_ConvertTo32Bits(imagen);
        
        if (!imagen) {
            return 0;
        }
        FreeImage_Unload(imagen);

        imagen = temp;
        //FreeImage_FlipVertical(imagen);

        width = FreeImage_GetWidth(imagen);
        height = FreeImage_GetHeight(imagen);

        image.resize(width * height * sizeof(uint32_t));
        memcpy(&image[0], swapBGRA8(FreeImage_GetBits(imagen), width * height), width * height * sizeof(uint32_t));

        return 0;
    }

#endif

    static int32_t tiled(int32_t sz, int32_t gmaxtile) {
        return (int32_t)ceil((double)sz / (double)gmaxtile);
    }

    static double milliseconds() {
        auto duration = std::chrono::high_resolution_clock::now();
        double millis = std::chrono::duration_cast<std::chrono::nanoseconds>(duration.time_since_epoch()).count() / 1000000.0;
        return millis;
    }

    static float frandom() {
        std::ranlux48 eng{ std::random_device{}() };
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(eng);
    }

    template<class T>
    size_t strided(size_t sizeo) {
        return sizeof(T) * sizeo;
    }



    static std::string readSource(const std::string &filePath, const bool& lineDirective = false) {
        std::string content;
        std::ifstream fileStream(filePath, std::ios::in);
        if (!fileStream.is_open()) {
            std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
            return "";
        }
        std::string line = "";
        while (!fileStream.eof()) {
            std::getline(fileStream, line);
            if (lineDirective || line.find("#line") == std::string::npos) content.append(line + "\n");
        }
        fileStream.close();
        return content;
    }

    static std::vector<char> readBinary(const std::string &filePath) {
        std::ifstream file(filePath, std::ios::in | std::ios::binary | std::ios::ate);
        std::vector<char> data;
        if (file.is_open()) {
            std::streampos size = file.tellg();
            data.resize(size);
            file.seekg(0, std::ios::beg);
            file.read(&data[0], size);
            file.close();
        }
        else {
            std::cerr << "Failure to open " + filePath << std::endl;
        }
        return data;
    };



}
