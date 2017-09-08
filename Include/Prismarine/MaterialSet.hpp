#pragma once

#include "Utils.hpp"
#include "Structs.hpp"
#include "TextureSet.hpp"

namespace NSM {
    class MaterialSet : public BaseClass {

    protected:
        
        TextureSet * texset = nullptr;
        GLuint mats = -1;
        std::vector<VirtualMaterial> submats;
        void init();

    public:

        MaterialSet() {init();}

        void setTextureSet(TextureSet *txs) { texset = txs; }
        void setTextureSet(TextureSet &txs) { texset = &txs; }
        void clearSubmats() { submats.resize(0); }

        size_t addSubmat(const VirtualMaterial * submat) ;
        size_t addSubmat(const VirtualMaterial &submat);
        void setSumbat(const size_t& i, const VirtualMaterial &submat);

        void loadToVGA();
        void bindWithContext(GLuint & prog);
    };
}
