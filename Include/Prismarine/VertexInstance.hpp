#pragma once

#include "Utils.hpp"
#include "Structs.hpp"

namespace ppr {

    class VertexInstance;

    class AccessorSet : public BaseClass {
    public:
        AccessorSet() {
            glCreateBuffers(1, &meshAccessorsBuffer);
        }
        friend SceneObject;
        friend VertexInstance;

        int32_t addVirtualAccessor(VirtualAccessor accessorDesc) {
            int32_t accessorPtr = meshAccessors.size();
            meshAccessors.push_back(accessorDesc);
            glNamedBufferData(meshAccessorsBuffer, meshAccessors.size() * sizeof(VirtualAccessor), meshAccessors.data(), GL_STATIC_DRAW);
            return accessorPtr;
        }

        void bind() {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, meshAccessorsBuffer != -1 ? meshAccessorsBuffer : 0);
        }

    private:
        std::vector<VirtualAccessor> meshAccessors;
        GLuint meshAccessorsBuffer = -1;
    };



    class BufferViewSet : public BaseClass {
    public:
        BufferViewSet() {
            glCreateBuffers(1, &bViewBuffer);
        }
        friend SceneObject;
        friend VertexInstance;

        int32_t addBufferView(VirtualBufferView accessorDesc) {
            int32_t accessorPtr = bufferViews.size();
            bufferViews.push_back(accessorDesc);
            glNamedBufferData(bViewBuffer, bufferViews.size() * sizeof(VirtualBufferView), bufferViews.data(), GL_STATIC_DRAW);
            return accessorPtr;
        }

        void bind() {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, bViewBuffer != -1 ? bViewBuffer : 0);
        }

    private:
        std::vector<VirtualBufferView> bufferViews;
        GLuint bViewBuffer = -1;
    };




    class VertexInstance : public BaseClass {
    public:
        VertexInstance() {
            GLuint dispatchData[3] = { 1, 1, 1 };
            glCreateBuffers(1, &indirect_dispatch_buffer);
            glNamedBufferData(indirect_dispatch_buffer, sizeof(dispatchData), dispatchData, GL_DYNAMIC_DRAW);

            glCreateBuffers(1, &meshUniformBuffer);
            glNamedBufferData(meshUniformBuffer, sizeof(MeshUniformStruct), &meshUniformData, GL_STATIC_DRAW);
        }
        friend SceneObject;

        size_t getNodeCount();
        void setNodeCount(size_t tcount);
        void setMaterialOffset(int32_t id);
        void useIndex16bit(bool b16);
        void setTransform(glm::mat4 t);
        void setTransform(glm::dmat4 t) {
            this->setTransform(glm::mat4(t));
        }

        void setIndexed(const int32_t b);
        void setVertices(const GLuint &buf);
        void setIndices(const GLuint &buf, const bool &all = true);
        void setLoadingOffset(const int32_t &off);

        void bind();


        // setting of accessors

        void setVertexAccessor(int32_t accessorID) {
            meshUniformData.vertexAccessor = accessorID;
            syncUniform();
        }

        void setNormalAccessor(int32_t accessorID) {
            meshUniformData.normalAccessor = accessorID;
            syncUniform();
        }

        void setTexcoordAccessor(int32_t accessorID) {
            meshUniformData.texcoordAccessor = accessorID;
            syncUniform();
        }

        void setModifierAccessor(int32_t accessorID) {
            meshUniformData.modifierAccessor = accessorID;
            syncUniform();
        }

        void setAccessorSet(AccessorSet * accessorSet) {
            this->accessorSet = accessorSet;
        }

        void setBufferViewSet(BufferViewSet * bufferViewSet) {
            this->bufferViewSet = bufferViewSet;
        }

    private:
        bool index16bit = false;

        GLuint vbo_triangle_ssbo = -1;
        GLuint mat_triangle_ssbo = -1;
        GLuint vebo_triangle_ssbo = -1;
        GLuint indirect_dispatch_buffer = -1;
        
        BufferViewSet * bufferViewSet;
        AccessorSet * accessorSet;
        MeshUniformStruct meshUniformData;
        GLuint meshUniformBuffer = -1;

        void syncUniform() {
            glNamedBufferData(meshUniformBuffer, sizeof(MeshUniformStruct), &meshUniformData, GL_STATIC_DRAW);
        }
    };
}

#include "./VertexInstance.inl"
