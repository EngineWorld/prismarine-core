#pragma once

#include "Utils.hpp"
#include "Structs.hpp"
#include "TextureSet.hpp"

namespace ppr {
    class MaterialSet : public BaseClass {

    private:
        
        TextureSet * texset = nullptr;
        GLuint mats = -1;
        std::vector<VirtualMaterial> submats;
        void init();

    public:

        MaterialSet() {
			submats = std::vector<VirtualMaterial>(0); // init
            init();
        }

        void setTextureSet(TextureSet *txs) { texset = txs; }
        void setTextureSet(TextureSet &txs) { texset = &txs; }
        void clearSubmats() { submats.resize(0); }

        size_t addSubmat(const VirtualMaterial * submat) {
            size_t idx = submats.size();
            submats.push_back(*submat);
            return idx;
        }

        size_t addSubmat(const VirtualMaterial &submat) {
            return this->addSubmat(&submat);
        }

        void setSumbat(const size_t& i, const VirtualMaterial &submat) {
            if (submats.size() <= i) submats.resize(i+1);
            submats[i] = submat;
        }

        void loadToVGA();
        void bindWithContext(GLuint & prog);
    };
}

#include "MaterialSet.inl"