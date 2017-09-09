#pragma once

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

#include "Utils.hpp"
#include "VertexInstance.hpp"
#include "Radix.hpp"

namespace NSM {

    class SceneObject : public BaseClass {
    protected:
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

        GLuint mat_triangle_ssbo_upload = -1;
        GLuint mat_triangle_ssbo = -1;

        GLuint vbo_sampler = -1;
        GLuint vbo_vertex_textrue = -1;
        GLuint vbo_normal_textrue = -1;
        GLuint vbo_texcoords_textrue = -1;
        GLuint vbo_modifiers_textrue = -1;

        GLuint vbo_vertex_textrue_upload = -1;
        GLuint vbo_normal_textrue_upload = -1;
        GLuint vbo_texcoords_textrue_upload = -1;
        GLuint vbo_modifiers_textrue_upload = -1;
        

        // uniform buffer
        GLuint geometryBlockUniform = -1;
        GeometryBlockUniform geometryBlockData;

        // uniforms
        GeometryUniformStruct geometryUniformData;


        GLuint aabbCounter = -1;
        GLuint leafBuffer = -1;
        GLuint bvhnodesBuffer = -1;
        GLuint mortonBuffer = -1;
        GLuint mortonBufferIndex = -1;
        GLuint bvhflagsBuffer = -1;
        GLuint activeBuffer = -1;
        GLuint childBuffer = -1;

        GLuint lscounterTemp = -1; // zero store
        GLuint minmaxBufRef = -1; // default bound

        GLuint tcounter = -1; // triangle counter
        GLuint minmaxBuf = -1; // minmax buffer

        void initShaders();
        void init();

        glm::vec3 offset = glm::vec3(-1.0001f);
        glm::vec3 scale  = glm::vec3( 2.0002f);

    public:

        SceneObject() { init(); }
        ~SceneObject();

        int32_t materialID = 0;
        size_t triangleCount = 0;

        void syncUniforms();
        void allocate(const size_t &count);
        void setMaterialID(int32_t id);
        void bindUniforms();
        void bind();
        void bindBVH();
        void bindLeafs();
        void clearTribuffer();
        void loadMesh(VertexInstance * gobject);
        bool isDirty() const;
        void markDirty();
        void resolve();
        void build(const glm::dmat4 &optimization = glm::dmat4(1.0));
        void configureIntersection(bool clearDepth);
    };
}
