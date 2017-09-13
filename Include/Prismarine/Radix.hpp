#pragma once

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

#include "Utils.hpp"
#include "Structs.hpp"

namespace NSM {

    class RadixSort {
        GLuint sortProgram = -1;
        GLuint permuteProgram = -1;
        GLuint histogramProgram = -1;
        GLuint pfxWorkProgram = -1;

        struct Consts { GLuint NumKeys, Shift, Descending, IsSigned; };
        const uint32_t WG_COUNT = 16;

        GLuint TmpKeys = -1;
        GLuint TmpValues = -1;
        GLuint VarBuffer = -1;
        GLuint Histograms = -1;
        GLuint PrefixSums = -1;

    public:

        RadixSort() {
            // for adopt for AMD
            initShaderComputeSPIRV("./shaders-spv/radix/single.comp.spv", sortProgram);
            initShaderComputeSPIRV("./shaders-spv/radix/permute.comp.spv", permuteProgram);
            initShaderComputeSPIRV("./shaders-spv/radix/histogram.comp.spv", histogramProgram);
            initShaderComputeSPIRV("./shaders-spv/radix/pfx-work.comp.spv", pfxWorkProgram);

            TmpKeys = allocateBuffer<uint64_t>(1024 * 1024 * 4);
            TmpValues = allocateBuffer<uint32_t>(1024 * 1024 * 4);
            Histograms = allocateBuffer<uint32_t>(WG_COUNT * 256);
            PrefixSums = allocateBuffer<uint32_t>(WG_COUNT * 256);
            VarBuffer = allocateBuffer<Consts>(1);
        }

        ~RadixSort() {
            glDeleteBuffers(1, &TmpKeys);
            glDeleteBuffers(1, &TmpValues);
            glDeleteBuffers(1, &VarBuffer);
        }

        void sort(GLuint &InKeys, GLuint &InVals, uint32_t size = 1, uint32_t descending = 0) {
            bool swapness = true;
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, InKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, InVals);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, VarBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 25, TmpKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 26, TmpValues);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 27, Histograms);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 28, PrefixSums);
            
            for (GLuint i = 0; i < 8; i++) { // 64-bit uint
			//for (GLuint i = 0; i < 4; i++) {
                Consts consts = { size, i, descending, 0 };
                glNamedBufferSubData(VarBuffer, 0, strided<Consts>(1), &consts);

                dispatch(histogramProgram, WG_COUNT);
                dispatch(pfxWorkProgram, 1);
                dispatch(permuteProgram, WG_COUNT);

                swapness = !swapness;
            }
        }

    };
    


}