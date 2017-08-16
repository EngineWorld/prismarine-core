#pragma once

#define RAY_TRACING_ENGINE

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <ctime>
#include <chrono>
#include <array>
#include <random>
#include <memory>
#include <sstream>
#include <map>
#include <algorithm>
#include <numeric>
#include <list>

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/component_wise.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"

#include "glbinding/Binding.h"
#include "glbinding/gl46ext/gl.h"

#ifdef USE_FREEIMAGE
#include "external/include/FreeImage.h"
#endif

namespace ppr {
    using namespace gl;
    
    class BaseClass {};
    class Dispatcher;
    class SceneObject;

    static int32_t tiled(int32_t sz, int32_t gmaxtile) {
        return (int32_t)ceil((double)sz / (double)gmaxtile);
    }

    static double milliseconds() {
        auto duration = std::chrono::high_resolution_clock::now();
        double millis = std::chrono::duration_cast<std::chrono::nanoseconds>(duration.time_since_epoch()).count() / 1000000.0;
        return millis;
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




    inline GLuint loadShaderSPIRV(std::string path, GLenum shaderType = GL_COMPUTE_SHADER, std::string entryName = "main") {
        std::vector<GLchar> str = readBinary(path);
        GLuint comp = glCreateShader(shaderType);
        {
            const GLchar * strc = str.data();
            int32_t size = str.size();
            glShaderBinary(1, &comp, GL_SHADER_BINARY_FORMAT_SPIR_V, strc, size);
            glSpecializeShader(comp, entryName.c_str(), 0, nullptr, nullptr);

            GLint status = false;
            glGetShaderiv(comp, GL_COMPILE_STATUS, &status);
            if (!status) {
                char * log = new char[32768];
                GLsizei len = 0;

                glGetShaderInfoLog(comp, 32768, &len, log);
                std::string logStr = std::string(log, len);
                std::cerr << logStr << std::endl;
            }
        }
        return comp;
    }




    inline void validateProgram(GLuint prog) {
        GLint status = false;
        glGetProgramiv(prog, GL_LINK_STATUS, &status);
        if (!status) {
            char * log = new char[32768];
            GLsizei len = 0;

            glGetProgramInfoLog(prog, 32768, &len, log);
            std::string logStr = std::string(log, len);
            std::cerr << logStr << std::endl;
        }
    }

    inline void initShaderComputeSPIRV(std::string path, GLuint & prog, std::string entryName = "main") {
        prog = glCreateProgram();
        glAttachShader(prog, loadShaderSPIRV(path, GL_COMPUTE_SHADER, entryName));
        glLinkProgram(prog);
        validateProgram(prog);
    }

    template<class T>
    GLuint allocateBuffer(size_t size = 1) {
        GLuint buf = 0;
        glCreateBuffers(1, &buf);
        glNamedBufferStorage(buf, sizeof(T) * size, nullptr, GL_DYNAMIC_STORAGE_BIT);
        return buf;
    }

    template<GLenum format = GL_RGBA8>
    GLuint allocateTexture2D(size_t width, size_t height) {
        GLuint tex = 0;
        glCreateTextures(GL_TEXTURE_2D, 1, &tex);
        glTextureStorage2D(tex, 1, format, width, height);
        return tex;
    }

    void dispatch(const GLuint &program, const GLuint gridSize) {
        glUseProgram(program);
        glDispatchCompute(gridSize, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    void dispatchIndirect(const GLuint &program, const GLuint& buffer) {
        glUseProgram(program);
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }


    const int32_t zero[1] = { 0 };

}
