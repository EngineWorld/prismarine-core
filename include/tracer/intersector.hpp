#pragma once

#include "includes.hpp"
#include "utils.hpp"
#include "mesh.hpp"
#include "radix.hpp"

namespace Paper {

    class Intersector : public PTObject {
    private:
        RadixSort * sorter = nullptr;

        bool dirty = false;
        uint32_t maxt = 1024 * 1024 * 1;
        uint32_t worksize = 128;

        GLuint geometryLoaderProgramI16 = -1;
        GLuint geometryLoaderProgram2 = -1;
        GLuint buildProgramH = -1;
        GLuint aabbMakerProgramH = -1;
        GLuint refitProgramH = -1;
        GLuint resortProgramH = -1;
        GLuint minmaxProgram2 = -1;

        GLuint mat_triangle_ssbo = -1;
        
        GLuint mat_triangle_ssbo_upload = -1;

        GLuint vbo_vertex_textrue = -1;
        GLuint vbo_normal_textrue = -1;
        GLuint vbo_texcoords_textrue = -1;
        GLuint vbo_modifiers_textrue = -1;

        GLuint vbo_vertex_textrue_upload = -1;
        GLuint vbo_normal_textrue_upload = -1;
        GLuint vbo_texcoords_textrue_upload = -1;
        GLuint vbo_modifiers_textrue_upload = -1;


        GLuint vbo_sampler = -1;

        // uniform buffer
        GLuint geometryBlockUniform = -1;
        GeometryBlockUniform geometryBlockData;

        // uniforms
        GeometryUniformStruct geometryUniformData;
        AttributeUniformStruct attributeUniformData;


        GLuint aabbCounter = -1;
        GLuint leafBuffer = -1;
        GLuint bvhnodesBuffer = -1;
        GLuint mortonBuffer = -1;
        GLuint mortonBufferIndex = -1;
        GLuint bvhflagsBuffer = -1;
        GLuint activeBuffer = -1;

        GLuint lscounterTemp = -1; // zero store
        GLuint minmaxBufRef = -1; // default bound

        GLuint tcounter = -1; // triangle counter
        GLuint minmaxBuf = -1; // minmax buffer

        void initShaders();
        void init();

        glm::vec3 offset = glm::vec3(-1.0001f);
        glm::vec3 scale  = glm::vec3( 2.0002f);

    public:

        Intersector() { init(); }
        ~Intersector() {
            glDeleteProgram(geometryLoaderProgramI16);
            glDeleteProgram(geometryLoaderProgram2);
            glDeleteProgram(buildProgramH);
            glDeleteProgram(aabbMakerProgramH);
            glDeleteProgram(refitProgramH);
            glDeleteProgram(resortProgramH);
            glDeleteProgram(minmaxProgram2);

            glDeleteBuffers(1, &mat_triangle_ssbo);
            glDeleteTextures(1, &vbo_vertex_textrue);
            glDeleteTextures(1, &vbo_normal_textrue);
            glDeleteTextures(1, &vbo_texcoords_textrue);
            glDeleteTextures(1, &vbo_modifiers_textrue);
            glDeleteSamplers(1, &vbo_sampler);

            glDeleteBuffers(1, &geometryBlockUniform);
            glDeleteBuffers(1, &aabbCounter);
            glDeleteBuffers(1, &leafBuffer);
            glDeleteBuffers(1, &bvhnodesBuffer);
            glDeleteBuffers(1, &mortonBuffer);
            glDeleteBuffers(1, &mortonBufferIndex);
            glDeleteBuffers(1, &bvhflagsBuffer);
            glDeleteBuffers(1, &lscounterTemp);
            glDeleteBuffers(1, &minmaxBufRef);
            glDeleteBuffers(1, &tcounter);
            glDeleteBuffers(1, &minmaxBuf);
        }

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
