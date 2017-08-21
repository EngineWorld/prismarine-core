#pragma once

#include "Utils.hpp"
#include "Structs.hpp"

namespace ppr {

    class RadixSort {
        GLuint sortProgram;

        struct Consts { GLuint NumKeys, Shift, Descending, IsSigned; };
        const uint32_t WG_COUNT = 1; // planned multiply radix sort support (aka. Async Compute)

        GLuint OutKeys = -1;
        GLuint OutValues = -1;
        GLuint VarBuffer = -1;

    public:

        RadixSort() {
            // for adopt for AMD
            initShaderComputeSPIRV("./shaders-spv/radix/single.comp.spv", sortProgram);

            OutKeys = allocateBuffer<uint64_t>(1024 * 1024 * 4);
            OutValues = allocateBuffer<uint32_t>(1024 * 1024 * 4);
            VarBuffer = allocateBuffer<Consts>(1);
        }

        ~RadixSort() {
            glDeleteBuffers(1, &OutKeys);
            glDeleteBuffers(1, &OutValues);
            glDeleteBuffers(1, &VarBuffer);
        }

        void sort(GLuint &InKeys, GLuint &InVals, uint32_t size = 1, uint32_t descending = 0) {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, InKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, InVals);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, OutKeys);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, OutValues);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, VarBuffer);

            //for (GLuint i = 0; i < 16; i++) { // 64-bit uint
            for (GLuint i = 0; i < 8; i++) { // 64-bit uint
                Consts consts = { size, i, descending, 0 };
                glNamedBufferSubData(VarBuffer, 0, strided<Consts>(1), &consts);
                dispatch(sortProgram, 1);
                glCopyNamedBufferSubData(OutKeys, InKeys, 0, 0, strided<uint64_t>(size));
                glCopyNamedBufferSubData(OutValues, InVals, 0, 0, strided<uint32_t>(size));
            }
        }

    };
    


}