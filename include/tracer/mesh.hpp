#pragma once

#include "includes.hpp"
#include "utils.hpp"

namespace Paper {
    class Mesh : public PTObject {
    public:
        Mesh() {
            GLuint dispatchData[3] = { 1, 1, 1 };
            glCreateBuffers(1, &indirect_dispatch_buffer);
            glNamedBufferData(indirect_dispatch_buffer, sizeof(dispatchData), dispatchData, GL_DYNAMIC_DRAW);

            glm::mat4 matrices[2] = { glm::transpose(trans), glm::transpose(glm::inverse(trans)) };
            glCreateBuffers(1, &transformBuffer);
            glNamedBufferData(transformBuffer, sizeof(glm::mat4)*2, matrices, GL_DYNAMIC_DRAW);
        }
        friend Intersector;

    private:
        GLuint vbo_triangle_ssbo = -1;
        GLuint mat_triangle_ssbo = -1;
        GLuint vebo_triangle_ssbo = -1;
        GLuint indirect_dispatch_buffer = -1;


        GLuint transformBuffer = -1;


        glm::mat4 texmat = glm::mat4(1.0f);
        glm::mat4 trans = glm::mat4(1.0f);
        glm::vec4 colormod = glm::vec4(1.0f);

        int32_t materialID = 0;
        int32_t unindexed = 1;
        int32_t offset = 0;
        size_t nodeCount = 0;
        float voffset = 0;
        bool index16bit = false;

    public:
        
        AttributeUniformStruct attributeUniformData;
        size_t getNodeCount();
        void setNodeCount(size_t tcount);
        void setVerticeOffset(float voff);
        void setColorModifier(glm::vec4 color);
        void setMaterialOffset(int32_t id);
        void useIndex16bit(bool b16);

        void setTransform(const glm::mat4 &t);
        void setTransformTexcoord(const glm::mat4 &t);

        void setIndexed(const int32_t b);
        void setVertices(const GLuint &buf);
        void setIndices(const GLuint &buf, const bool &all = true);
        void setLoadingOffset(const int32_t &off);

        void bind();
    };
}

#include "./mesh.inl"
