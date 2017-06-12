#pragma once

#include "includes.hpp"
#include "utils.hpp"

//#define TOL_SUPPORT

#ifdef TOL_SUPPORT
#include "tiny_obj_loader/tiny_obj_loader.h"
#endif

#ifdef ASSIMP_SUPPORT
#include "assimp/Importer.hpp"
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>     // Post processing flags
#include <assimp/texture.h>
#endif

namespace Paper {
    class Mesh : public PTObject {
    public:
        Mesh() {}
        friend Intersector;

    private:
        GLuint vbo_triangle_ssbo = -1;
        GLuint mat_triangle_ssbo = -1;
        GLuint vebo_triangle_ssbo = -1;
        int32_t _toffset = 0;

        AttributeUniformStruct attributeUniformData;
        glm::mat4 texmat = glm::mat4(1.0f);
        glm::mat4 trans = glm::mat4(1.0f);
        glm::vec4 colormod = glm::vec4(1.0f);

        int32_t materialID = 0;
        int32_t maxDepth = 4;
        int32_t unindexed = 1;
        int32_t offset = 0;
        size_t nodeCount = 0;
        size_t verticeCount = 0;
        
        float voffset = 0;


    public:
        
        size_t getNodeCount();
        void setNodeCount(size_t tcount);
        void setVerticeOffset(float voff);
        void setColorModifier(glm::vec4 color);
        void setMaterialOffset(int32_t id);

        void setTransform(const glm::mat4 &t);
        void setTransformTexcoord(const glm::mat4 &t);

        void setIndexed(const int32_t b);
        void setVertices(const GLuint &buf);
        void setIndices(const GLuint &buf, const bool &all = true);
        void setLoadingOffset(const int32_t &off);

#ifdef ASSIMP_SUPPORT
        void loadMesh(aiMesh ** meshes, int32_t meshCount);
#endif

        void bind();
    };
}

#include "./mesh.inl"
