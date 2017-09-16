#pragma once

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

#include "Utils.hpp"
#include "Structs.hpp"
#include "TextureSet.hpp"

namespace NSM {
    class TriangleHierarchy;
    class Pipeline;

    class MaterialSet : public BaseClass {

    protected:

        friend class Pipeline;
        friend class TriangleHierarchy;
        TextureSet * texset = nullptr;
        GLuint mats = -1;
        std::vector<VirtualMaterial> submats;
        void init();
        GLuint countBuffer = -1;
        GLint loadOffset = 0;

    public:

        MaterialSet() { init(); }

        void setTextureSet(TextureSet *txs) { texset = txs; }
        void setTextureSet(TextureSet &txs) { texset = &txs; }
        void clearSubmats() { submats.resize(0); }

        size_t getMaterialCount();
        size_t addSubmat(const VirtualMaterial * submat);
        size_t addSubmat(const VirtualMaterial &submat);
        void setSumbat(const size_t& i, const VirtualMaterial &submat);

        void loadToVGA();
        void bindWithContext(GLuint & prog);
        void setLoadingOffset(GLint loadOffset);
    };
}
