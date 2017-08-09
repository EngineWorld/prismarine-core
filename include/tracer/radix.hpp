#pragma once

namespace ppr {

    class RadixSort {
        GLuint permuteProgram;
        GLuint prefixScanProgram;
        GLuint histogramProgram;
        //GLuint flipProgram;

        struct Consts { GLuint NumKeys, Shift, Descending, IsSigned; };
        const uint32_t WG_COUNT = 8;
        const uint32_t RADICES = 16;

        GLuint OutKeys;
        GLuint OutValues;
        GLuint HistogramBuffer;
        GLuint VarBuffer;

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

             OutKeys = allocateBuffer<uint64_t>(1024 * 1024 * 4);
             OutValues = allocateBuffer<uint32_t>(1024 * 1024 * 4);
             HistogramBuffer = allocateBuffer<uint32_t>(WG_COUNT * RADICES);
             VarBuffer = allocateBuffer<Consts>(1);
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
                { size, 28, descending, 0 },
                { size, 32, descending, 0 },
                { size, 36, descending, 0 },
                { size, 40, descending, 0 },
                { size, 44, descending, 0 },
                { size, 48, descending, 0 },
                { size, 52, descending, 0 },
                { size, 56, descending, 0 },
                { size, 60, descending, 0 }
            };

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, InKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, InVals);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, OutKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, OutValues);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, VarBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 25, HistogramBuffer);

            //for (GLuint i = 0; i < 8;i++) { // 32-bit uint
            for (GLuint i = 0; i < 16; i++) { // 64-bit uint
                glNamedBufferSubData(VarBuffer, 0, strided<Consts>(1), &consts[i]);
                dispatch(histogramProgram, WG_COUNT);
                dispatch(prefixScanProgram, 1);
                dispatch(permuteProgram, WG_COUNT);
                glCopyNamedBufferSubData(OutKeys, InKeys, 0, 0, strided<uint64_t>(size));
                //glCopyNamedBufferSubData(OutKeys, InKeys, 0, 0, strided<uint32_t>(size));
                glCopyNamedBufferSubData(OutValues, InVals, 0, 0, strided<uint32_t>(size));
            }
        }

    };
    


}