#pragma once

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

#define RAY_TRACING_ENGINE

#include <vector>
#include <iostream>
#include <chrono>
#include <array>
#include <random>
#include <map>

#include "GL/glew.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/component_wise.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/gtx/transform.hpp"

#ifdef USE_FREEIMAGE
#include "external/include/FreeImage.h"
#endif

//#define PROFILE_RT
#define NSM psm

namespace NSM {
    //using namespace gl;

    class BaseClass {};
    class Pipeline;
    class TriangleHierarchy;

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

#ifdef USE_OPENGL_45_COMPATIBLE
            glShaderBinary(1, &comp, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB, strc, size);
            glSpecializeShaderARB(comp, entryName.c_str(), 0, nullptr, nullptr);
#else
            glShaderBinary(1, &comp, GL_SHADER_BINARY_FORMAT_SPIR_V, strc, size);
            glSpecializeShader(comp, entryName.c_str(), 0, nullptr, nullptr);
#endif

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
        glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        return tex;
    }

    void dispatch(const GLuint &program, const GLuint gridSize) {
        glUseProgram(program);
        glDispatchCompute(gridSize, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        //#ifdef PROFILE_RT
        //		glFinish();
        //#endif
    }

    void dispatchIndirect(const GLuint &program, const GLuint& buffer) {
        glUseProgram(program);
        glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, buffer);
        glDispatchComputeIndirect(0);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    void SWAP(GLuint& buf1, GLuint& buf2) {
        GLuint tmp = buf1;
        buf1 = buf2;
        buf2 = tmp;
    }

    const int32_t zero[1] = { 0 };

}
