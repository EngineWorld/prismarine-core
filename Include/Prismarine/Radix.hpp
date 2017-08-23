#pragma once

#include "Utils.hpp"
#include "Structs.hpp"

namespace ppr {

    class RadixSort {
        GLuint sortProgram;

        struct Consts { GLuint NumKeys, Shift, Descending, IsSigned; };
        const uint32_t WG_COUNT = 2; // planned multiply radix sort support (aka. Async Compute)

        GLuint OutKeys = -1;
        GLuint OutValues = -1;
        GLuint TmpKeys = -1;
        GLuint TmpValues = -1;
        GLuint VarBuffer = -1;
        GLuint Histograms = -1;

    public:

        RadixSort() {
            // for adopt for AMD
            initShaderComputeSPIRV("./shaders-spv/radix/single.comp.spv", sortProgram);

            OutKeys = allocateBuffer<uint64_t>(1024 * 1024 * 4);
            OutValues = allocateBuffer<uint32_t>(1024 * 1024 * 4);
            TmpKeys = allocateBuffer<uint64_t>(1024 * 1024 * 4);
            TmpValues = allocateBuffer<uint32_t>(1024 * 1024 * 4);
            Histograms = allocateBuffer<uint32_t>(WG_COUNT * 256);
            VarBuffer = allocateBuffer<Consts>(1);
        }

        ~RadixSort() {
            glDeleteBuffers(1, &OutKeys);
            glDeleteBuffers(1, &OutValues);
            glDeleteBuffers(1, &VarBuffer);
        }

        void sort(GLuint &InKeys, GLuint &InVals, uint32_t size = 1, uint32_t descending = 0) {
            bool swapness = true;
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, VarBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 25, TmpKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 26, TmpValues);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 27, Histograms);

            for (GLuint i = 0; i < 8; i++) { // 64-bit uint
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, swapness ? InKeys : OutKeys);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, swapness ? InVals : OutValues);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, swapness ? OutKeys : InKeys);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, swapness ? OutValues : InVals);
                

                Consts consts = { size, i, descending, 0 };
                glNamedBufferSubData(VarBuffer, 0, strided<Consts>(1), &consts);
                dispatch(sortProgram, 1);
                swapness = !swapness;
            }

            if (!swapness) glCopyNamedBufferSubData(OutKeys, InKeys, 0, 0, strided<uint64_t>(size));
            if (!swapness) glCopyNamedBufferSubData(OutValues, InVals, 0, 0, strided<uint32_t>(size));
        }

    };
    


}