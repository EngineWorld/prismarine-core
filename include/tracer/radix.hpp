#pragma once

namespace Paper {

    class RadixSort {
        GLuint permuteProgram;
        GLuint prefixScanProgram;
        GLuint histogramProgram;
        //GLuint flipProgram;

        struct Consts { GLuint NumKeys, Shift, Descending, IsSigned; };

    public:

        RadixSort() {
            /*
            initShaderCompute("./shaders/radix/permute.comp", permuteProgram);
            initShaderCompute("./shaders/radix/prefix-scan.comp", prefixScanProgram);
            initShaderCompute("./shaders/radix/histogram.comp", histogramProgram);
            */
            initShaderComputeSPIRV("./shaders-spv/radix/permute.comp.spv", permuteProgram);
            initShaderComputeSPIRV("./shaders-spv/radix/prefix-scan.comp.spv", prefixScanProgram);
            initShaderComputeSPIRV("./shaders-spv/radix/histogram.comp.spv", histogramProgram);
        }

        void sort(GLuint &InKeys, GLuint &InVals, uint32_t size = 1, uint32_t descending = 0) {
            Consts consts[] = {
                { size, 0, descending, 0 },
                { size, 4, descending, 0 },
                { size, 8, descending, 0 },
                { size, 12, descending, 0 },
                { size, 16, descending, 0 },
                { size, 20, descending, 0 },
                { size, 24, descending, 0 },
                { size, 28, descending, 0 }
            };

            const uint32_t WG_COUNT = 8;
            const uint32_t RADICES = 16;

            GLuint OutKeys = allocateBuffer<uint32_t>(size);
            GLuint OutValues = allocateBuffer<uint32_t>(size);
            GLuint HistogramBuffer = allocateBuffer<uint32_t>(WG_COUNT * RADICES);
            GLuint VarBuffer = allocateBuffer<Consts>(1);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, InKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, InVals);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, OutKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, OutValues);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, VarBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 25, HistogramBuffer);

            for (GLuint i = 0; i < 8;i++) {
                glNamedBufferSubData(VarBuffer, 0, strided<Consts>(1), &consts[i]);
                dispatch(histogramProgram, WG_COUNT);
                dispatch(prefixScanProgram, 1);
                dispatch(permuteProgram, WG_COUNT);
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