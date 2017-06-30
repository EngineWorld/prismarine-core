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
        uint32_t worksize = 64;

        GLuint geometryLoaderProgramI16 = -1;
        GLuint geometryLoaderProgram2 = -1;
        GLuint buildProgramH = -1;
        GLuint aabbMakerProgramH = -1;
        GLuint refitProgramH = -1;
        GLuint resortProgramH = -1;
        GLuint minmaxProgram2 = -1;

        GLuint ebo_triangle_ssbo = -1;
        GLuint mat_triangle_ssbo = -1;
        GLuint vbo_triangle_ssbo = -1;
        GLuint geometryBlockUniform = -1;

        GeometryBlockUniform geometryBlockData;
        MinmaxUniformStruct minmaxUniformData;
        OctreeUniformStruct octreeUniformData;
        GeometryUniformStruct geometryUniformData;
        AttributeUniformStruct attributeUniformData;

        GLuint aabbCounter = -1;
        GLuint nodeCounter = -1;
        GLuint numBuffer = -1;
        GLuint leafBuffer = -1;
        GLuint leafBufferSorted = -1;
        GLuint bvhnodesBuffer = -1;
        GLuint mortonBuffer = -1;
        GLuint mortonBufferIndex = -1;

        GLuint bvhflagsBuffer = -1;
        GLuint lscounterTemp = -1;
        GLuint tcounter = -1;
        GLuint minmaxBuf = -1;
        GLuint minmaxBufRef = -1;

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
        bbox bound;

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
        void build(const glm::dmat4 &optimization = glm::dmat4(1.0));
        void configureIntersection(bool clearDepth);
    };
}

#include "./intersector.inl"
