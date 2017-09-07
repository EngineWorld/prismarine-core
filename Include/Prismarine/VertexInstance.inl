#include "VertexInstance.hpp"

namespace ppr {



    template<int BINDING, class STRUCTURE>
    inline int32_t BufferComposer<BINDING, STRUCTURE>::addElement(STRUCTURE element) {
        int32_t ptr = data.size();
        data.push_back(element);
        glNamedBufferData(buffer, data.size() * sizeof(STRUCTURE), data.data(), GL_STATIC_DRAW);
        return ptr;
    }

    template<int BINDING, class STRUCTURE>
    inline void BufferComposer<BINDING, STRUCTURE>::bind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING, buffer != -1 ? buffer : 0);
    }




    inline VertexInstance::VertexInstance() {
        GLuint dispatchData[3] = { 1, 1, 1 };
        glCreateBuffers(1, &indirect_dispatch_buffer);
        glNamedBufferData(indirect_dispatch_buffer, sizeof(dispatchData), dispatchData, GL_DYNAMIC_DRAW);

        glCreateBuffers(1, &meshUniformBuffer);
        glNamedBufferData(meshUniformBuffer, sizeof(MeshUniformStruct), &meshUniformData, GL_STATIC_DRAW);
    }

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

    inline void VertexInstance::syncUniform() {
        glNamedBufferData(meshUniformBuffer, sizeof(MeshUniformStruct), &meshUniformData, GL_STATIC_DRAW);
    }


    // setting of accessors

    inline void VertexInstance::setVertexAccessor(int32_t accessorID) {
        meshUniformData.vertexAccessor = accessorID;
        syncUniform();
    }

    inline void VertexInstance::setNormalAccessor(int32_t accessorID) {
        meshUniformData.normalAccessor = accessorID;
        syncUniform();
    }

    inline void VertexInstance::setTexcoordAccessor(int32_t accessorID) {
        meshUniformData.texcoordAccessor = accessorID;
        syncUniform();
    }

    inline void VertexInstance::setModifierAccessor(int32_t accessorID) {
        meshUniformData.modifierAccessor = accessorID;
        syncUniform();
    }

    inline void VertexInstance::setAccessorSet(AccessorSet * accessorSet) {
        this->accessorSet = accessorSet;
    }

    inline void VertexInstance::setBufferViewSet(BufferViewSet * bufferViewSet) {
        this->bufferViewSet = bufferViewSet;
    }

}
