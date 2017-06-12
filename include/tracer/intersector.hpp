#pragma once

#include "includes.hpp"
#include "utils.hpp"
#include "mesh.hpp"
#include "radix.hpp"

namespace Paper {

    class Intersector : public PTObject {
    private:
        RadixSort * sorter;

        const int32_t zero[1] = { 0 };
        bool dirty = false;
        uint32_t maxt = 1024 * 1024 * 1;
        uint32_t worksize = 256;

        GLuint geometryLoaderProgram2;
        GLuint buildProgramH;
        GLuint aabbMakerProgramH;
        GLuint refitProgramH;
        GLuint resortProgramH;
        GLuint minmaxProgram2;

        GLuint ebo_triangle_ssbo;
        GLuint mat_triangle_ssbo;
        GLuint vbo_triangle_ssbo;
        GLuint geometryBlockUniform;
        GeometryBlockUniform geometryBlockData;
        MinmaxUniformStruct minmaxUniformData;
        OctreeUniformStruct octreeUniformData;
        GeometryUniformStruct geometryUniformData;
        AttributeUniformStruct attributeUniformData;

        GLuint aabbCounter;
        GLuint nodeCounter;
        GLuint numBuffer;
        GLuint leafBuffer;
        GLuint leafBufferSorted;
        GLuint bvhnodesBuffer;
        GLuint mortonBuffer;
        GLuint mortonBufferIndex;

        GLuint bvhflagsBuffer;
        GLuint lscounterTemp;
        GLuint tcounter;
        GLuint minmaxBuf;
        GLuint minmaxBufRef;

        void initShaderCompute(std::string str, GLuint& prog);
        void initShaders();
        void init();

        glm::vec3 offset = glm::vec3(-1.0001f);
        glm::vec3 scale  = glm::vec3( 2.0002f);

    public:

        Intersector() { init(); }

        int32_t materialID = 0;
        size_t triangleCount = 0;
        size_t verticeCount = 0;

        void syncUniforms();
        void allocate(const size_t &count);
        void setMaterialID(int32_t id);
        void bindUniforms();
        void bind();
        void bindBVH();
        void clearTribuffer();
        void loadMesh(Mesh * gobject);
        bool isDirty() const;
        void markDirty();
        void resolve();
        void build(const glm::mat4 &optimization = glm::mat4(1.0f));
        void configureIntersection(bool clearDepth);
    };
}

#include "./intersector.inl"
