#pragma once

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

#include "Utils.hpp"
#include "Structs.hpp"

namespace NSM {

    class VertexInstance;

    template<int BINDING, class STRUCTURE>
    class BufferComposer : public BaseClass {
    public:
        BufferComposer() {
            glCreateBuffers(1, &buffer);
        }
        friend VertexInstance;

        int32_t addElement(STRUCTURE accessorDesc);
        void bind();

    protected:
        GLuint buffer = -1;
        std::vector<STRUCTURE> data;
    };

    using AccessorSet = BufferComposer<7, VirtualAccessor>;
    using BufferViewSet = BufferComposer<8, VirtualBufferView>;



    class VertexInstance : public BaseClass {
    public:
    };

    class TriangleArrayInstance : public VertexInstance {
    public:
        TriangleArrayInstance();
        friend TriangleHierarchy;

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
        void setVertexAccessor(int32_t accessorID);
        void setNormalAccessor(int32_t accessorID);
        void setTexcoordAccessor(int32_t accessorID);
        void setModifierAccessor(int32_t accessorID);
        void setAccessorSet(AccessorSet * accessorSet);
        void setBufferViewSet(BufferViewSet * bufferViewSet);

    protected:
        bool index16bit = false;

        GLuint vbo_triangle_ssbo = -1;
        GLuint mat_triangle_ssbo = -1;
        GLuint vebo_triangle_ssbo = -1;
        GLuint indirect_dispatch_buffer = -1;

        BufferViewSet * bufferViewSet;
        AccessorSet * accessorSet;
        MeshUniformStruct meshUniformData;
        GLuint meshUniformBuffer = -1;

        void syncUniform();
    };
}
