#pragma once

namespace Paper {

    class RadixSort {
        GLuint permuteProgram;
        GLuint prefixScanProgram;
        GLuint histogramProgram;
        GLuint flipProgram;

        struct Consts { GLuint NumKeys, Shift, Descending, IsSigned; };

        void initShaderCompute(std::string path, GLuint& prog) {
            std::string str = readSource(path);

            GLuint comp = glCreateShader(GL_COMPUTE_SHADER);
            {
                const char * strc = str.c_str();
                int32_t size = str.size();
                glShaderSource(comp, 1, &strc, &size);
                glCompileShader(comp);

                GLint status = false;
                glGetShaderiv(comp, GL_COMPILE_STATUS, &status);
                if (!status) {
                    char * log = new char[1024];
                    GLsizei len = 0;

                    glGetShaderInfoLog(comp, 1024, &len, log);
                    std::string logStr = std::string(log, len);
                    std::cerr << logStr << std::endl;
                }
            }

            prog = glCreateProgram();
            glAttachShader(prog, comp);
            glLinkProgram(prog);

            GLint status = false;
            glGetProgramiv(prog, GL_LINK_STATUS, &status);
            if (!status) {
                char * log = new char[1024];
                GLsizei len = 0;

                glGetProgramInfoLog(prog, 1024, &len, log);
                std::string logStr = std::string(log, len);
                std::cerr << logStr << std::endl;
            }
        }


        template<class T>
        GLuint allocateBuffer(size_t size = 1) {
            GLuint buf = 0; 
            glCreateBuffers(1, &buf); 
            glNamedBufferStorage(buf, sizeof(T) * size, nullptr, GL_DYNAMIC_STORAGE_BIT); 
            return buf;
        }

        void dispatch(const GLuint &program, const GLuint gridSize) {
            glUseProgram(program);
            glDispatchCompute(gridSize, 1, 1);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
        }



    public:

        RadixSort() {
            initShaderCompute("./shaders/radix/permute.comp", permuteProgram);
            initShaderCompute("./shaders/radix/prefix-scan.comp", prefixScanProgram);
            initShaderCompute("./shaders/radix/histogram.comp", histogramProgram);
        }

        void sort(GLuint &InKeys, GLuint &InVals, uint32_t size = 1, uint32_t descending = 0) {
            Consts consts[] = {
                size, 0, descending, 0,
                size, 4, descending, 0,
                size, 8, descending, 0,
                size, 12, descending, 0,
                size, 16, descending, 0,
                size, 20, descending, 0,
                size, 24, descending, 0,
                size, 28, descending, 0
            };

            const uint32_t WG_COUNT = 64;
            const uint32_t RADICES = 16;

            GLuint OutKeys = this->allocateBuffer<uint32_t>(size);
            GLuint OutValues = this->allocateBuffer<uint32_t>(size);
            GLuint HistogramBuffer = this->allocateBuffer<uint32_t>(WG_COUNT * RADICES);
            GLuint VarBuffer = this->allocateBuffer<Consts>(1);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, InKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, InVals);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, OutKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, OutValues);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, VarBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, HistogramBuffer);

            for (GLuint i = 0; i < 8;i++) {
                glNamedBufferSubData(VarBuffer, 0, strided<Consts>(1), &consts[i]);
                this->dispatch(histogramProgram, WG_COUNT);
                this->dispatch(prefixScanProgram, 1);
                this->dispatch(permuteProgram, WG_COUNT);
                glCopyNamedBufferSubData(OutKeys, InKeys, 0, 0, strided<uint32_t>(size));
                glCopyNamedBufferSubData(OutValues, InVals, 0, 0, strided<uint32_t>(size));
            }

            glFlush();
            glDeleteBuffers(1, &OutKeys);
            glDeleteBuffers(1, &OutValues);
            glDeleteBuffers(1, &HistogramBuffer);
            glDeleteBuffers(1, &VarBuffer);
        }

    };
    


}