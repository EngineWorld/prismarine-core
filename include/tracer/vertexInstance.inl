#include "vertexInstance.hpp"

namespace ppr {

    inline void VertexInstance::setNodeCount(size_t tcount) {
        uint32_t tiledWork = tiled(tcount, 128);
        glNamedBufferSubData(indirect_dispatch_buffer, 0, sizeof(uint32_t), &tiledWork);
        meshUniformData.nodeCount = tcount;
        syncUniform();
    }

    inline size_t VertexInstance::getNodeCount() {
        return meshUniformData.nodeCount;
    }

    inline void VertexInstance::useIndex16bit(bool b16) {
        index16bit = b16;
    }

    inline void VertexInstance::setMaterialOffset(int32_t id) {
        meshUniformData.materialID = id;
        syncUniform();
    }

    inline void VertexInstance::setTransform(glm::mat4 t) {
        meshUniformData.transform = glm::transpose(t);
        meshUniformData.transformInv = glm::inverse(t);
        syncUniform();
    }

    inline void VertexInstance::setIndexed(const int32_t b) {
        meshUniformData.isIndexed = b;
        syncUniform();
    }

    inline void VertexInstance::setLoadingOffset(const int32_t &off) {
        meshUniformData.loadingOffset = off;
        syncUniform();
    }

    inline void VertexInstance::setVertices(const GLuint &buf) {
        vbo_triangle_ssbo = buf;
    }

    inline void VertexInstance::setIndices(const GLuint &buf, const bool &all) {
        vebo_triangle_ssbo = buf;
    }

    inline void VertexInstance::bind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vbo_triangle_ssbo != -1 ? vbo_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, vebo_triangle_ssbo != -1 ? vebo_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mat_triangle_ssbo != -1 ? mat_triangle_ssbo : 0);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, meshUniformBuffer != -1 ? meshUniformBuffer : 0);
    }

}
