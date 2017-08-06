#include "mesh.hpp"

namespace Paper {

    inline void Mesh::setNodeCount(size_t tcount) {
        uint32_t tiledWork = tiled(tcount, 128);
        glNamedBufferSubData(indirect_dispatch_buffer, 0, sizeof(uint32_t), &tiledWork);
        nodeCount = tcount;
    }

    inline size_t Mesh::getNodeCount() {
        return nodeCount;
    }

    inline void Mesh::useIndex16bit(bool b16) {
        index16bit = b16;
    }

    inline void Mesh::setVerticeOffset(float voff) {
        voffset = voff;
    }

    inline void Mesh::setColorModifier(glm::vec4 color) {
        colormod = color;
    }

    inline void Mesh::setMaterialOffset(int32_t id) {
        materialID = id;
    }

    inline void Mesh::setTransform(const glm::mat4 &t) {
        trans = t;

        glm::mat4 matrices[2] = { glm::transpose(trans), glm::inverse(trans) };
        glNamedBufferData(transformBuffer, sizeof(glm::mat4) * 2, matrices, GL_DYNAMIC_DRAW);
    }

    inline void Mesh::setTransformTexcoord(const glm::mat4 &t) {
        texmat = t;
    }

    inline void Mesh::setIndexed(const int32_t b) {
        unindexed = b == 0 ? 1 : 0;
    }

    inline void Mesh::setVertices(const GLuint &buf) {
        vbo_triangle_ssbo = buf;
    }

    inline void Mesh::setIndices(const GLuint &buf, const bool &all) {
        vebo_triangle_ssbo = buf;
    }

    inline void Mesh::setLoadingOffset(const int32_t &off) {
        offset = off;
    }

    inline void Mesh::bind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo_triangle_ssbo != -1 ? vbo_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vebo_triangle_ssbo != -1 ? vebo_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mat_triangle_ssbo != -1 ? mat_triangle_ssbo : 0);
    }

}
