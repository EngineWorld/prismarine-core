#include "mesh.hpp"

namespace Paper {

    inline void Mesh::setNodeCount(size_t tcount) {
        uint32_t tiledWork = tiled(tcount, 128);
        glNamedBufferSubData(indirect_dispatch_buffer, 0, sizeof(uint32_t), &tiledWork);
        meshUniformData.nodeCount = tcount;
        syncUniform();
    }

    inline size_t Mesh::getNodeCount() {
        return meshUniformData.nodeCount;
    }

    inline void Mesh::useIndex16bit(bool b16) {
        index16bit = b16;
    }

    inline void Mesh::setMaterialOffset(int32_t id) {
        meshUniformData.materialID = id;
        syncUniform();
    }

    inline void Mesh::setTransform(glm::mat4 t) {
        meshUniformData.transform = glm::transpose(t);
        meshUniformData.transformInv = glm::inverse(t);
        syncUniform();
    }

    inline void Mesh::setIndexed(const int32_t b) {
        meshUniformData.isIndexed = b;
        syncUniform();
    }

    inline void Mesh::setLoadingOffset(const int32_t &off) {
        meshUniformData.loadingOffset = off;
        syncUniform();
    }

    inline void Mesh::setVertices(const GLuint &buf) {
        vbo_triangle_ssbo = buf;
    }

    inline void Mesh::setIndices(const GLuint &buf, const bool &all) {
        vebo_triangle_ssbo = buf;
    }

    inline void Mesh::bind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo_triangle_ssbo != -1 ? vbo_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vebo_triangle_ssbo != -1 ? vebo_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mat_triangle_ssbo != -1 ? mat_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, meshUniformBuffer != -1 ? meshUniformBuffer : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, meshAccessorsBuffer != -1 ? meshAccessorsBuffer : 0);
    }

}
