#include "mesh.hpp"

namespace Paper {

    inline void Mesh::setNodeCount(size_t tcount) {
        nodeCount = tcount;
    }

    inline size_t Mesh::getNodeCount() {
        return nodeCount;
    }

    inline void Mesh::useIndex16bit(bool b16) {
        index16bit = b16;
    }

    inline void Mesh::setVerticeOffset(float voff) {
        voffset = voff;
    }

    inline void Mesh::setColorModifier(glm::vec4 color) {
        colormod = color;
    }

    inline void Mesh::setMaterialOffset(int32_t id) {
        materialID = id;
    }

    inline void Mesh::setTransform(const glm::mat4 &t) {
        trans = t;
    }

    inline void Mesh::setTransformTexcoord(const glm::mat4 &t) {
        texmat = t;
    }

    inline void Mesh::setIndexed(const int32_t b) {
        unindexed = b == 0 ? 1 : 0;
    }

    inline void Mesh::setVertices(const GLuint &buf) {
        vbo_triangle_ssbo = buf;
    }

    inline void Mesh::setIndices(const GLuint &buf, const bool &all) {
        vebo_triangle_ssbo = buf;
    }

    inline void Mesh::setLoadingOffset(const int32_t &off) {
        offset = off;
    }

#ifdef ASSIMP_SUPPORT
    inline void Mesh::loadMesh(aiMesh ** meshes, int32_t meshCount) {
        std::vector<int32_t> vindices(0);
        std::vector<int32_t> material_ids(0);
        std::vector<float> stridedData(0);

        size_t stride = 8;
        size_t tcount = 0;
        for (int32_t i = 0; i < meshCount; i++) {
            aiMesh * mesh = meshes[i];

            for (uint32_t ti = 0; ti < mesh->mNumFaces;ti++) {
                aiFace &face = mesh->mFaces[ti];
                material_ids.push_back(mesh->mMaterialIndex);
                //for (int32_t dc = 0; dc < face.mNumIndices;dc++) {
                for (int32_t dc = 0; dc < 3; dc++) {
                    vindices.push_back(_toffset + face.mIndices[dc]);
                }
                tcount++;
            }

            for (uint32_t ti = 0; ti < mesh->mNumVertices; ti++) {
                aiVector3D &vertice = mesh->mVertices[ti];
                stridedData.push_back(vertice.x);
                stridedData.push_back(vertice.y);
                stridedData.push_back(vertice.z);

                if (mesh->HasTextureCoords(0) && mesh->mTextureCoords[0]) {
                    aiVector3D &texcoord = mesh->mTextureCoords[0][ti];
                    stridedData.push_back(texcoord.x);
                    stridedData.push_back(texcoord.y);
                } else {
                    stridedData.push_back(0);
                    stridedData.push_back(0);
                }

                if (mesh->HasNormals() && mesh->mNormals) {
                    aiVector3D &normal = mesh->mNormals[ti];
                    stridedData.push_back(normal.x);
                    stridedData.push_back(normal.y);
                    stridedData.push_back(normal.z);
                } else {
                    stridedData.push_back(0);
                    stridedData.push_back(0);
                    stridedData.push_back(0);
                }
            }

            _toffset += mesh->mNumVertices;
        }

        unindexed = 0;
        nodeCount = tcount;
        verticeCount = stridedData.size() / stride;

        // make owner layout template
        attributeUniformData.mode = 0;
        attributeUniformData.stride = stride;
        attributeUniformData.haveTexcoord = true;
        attributeUniformData.haveNormal = true;
        attributeUniformData.haveColor = false;
        attributeUniformData.vertexOffset = 0;
        attributeUniformData.texcoordOffset = 3;
        attributeUniformData.normalOffset = 5;

        // delete not needed buffers
        if (vbo_triangle_ssbo != -1) glDeleteBuffers(1, &vbo_triangle_ssbo);
        if (mat_triangle_ssbo != -1) glDeleteBuffers(1, &mat_triangle_ssbo);
        if (vebo_triangle_ssbo != -1) glDeleteBuffers(1, &vebo_triangle_ssbo);

        // re-alloc buffers
        glCreateBuffers(1, &vbo_triangle_ssbo);
        glCreateBuffers(1, &mat_triangle_ssbo);
        glCreateBuffers(1, &vebo_triangle_ssbo);

        glNamedBufferData(vebo_triangle_ssbo, strided<int32_t>(vindices.size()), vindices.data(), GL_STATIC_DRAW);
        glNamedBufferData(mat_triangle_ssbo, strided<int32_t>(material_ids.size()), material_ids.data(), GL_STATIC_DRAW);
        glNamedBufferData(vbo_triangle_ssbo, strided<float>(stridedData.size()), stridedData.data(), GL_STATIC_DRAW);
    }
#endif

    inline void Mesh::bind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, vbo_triangle_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, vebo_triangle_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, mat_triangle_ssbo);
    }

}
