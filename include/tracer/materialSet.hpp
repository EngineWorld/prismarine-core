#pragma once

#include "includes.hpp"
#include "utils.hpp"
#include <map>
#include <algorithm>

namespace ppr {
    class MaterialSet : public BaseClass {

    private:
        //int32_t materialID = 0;
        GLuint mats = -1;
        GLuint texturesBuffer = -1;
        std::vector<uint64_t> vctr;

        void init();

    public:

        std::vector<Material> submats;
        std::vector<uint32_t> textures;
        std::vector<uint32_t> freedomTextures;
        std::map<std::string, uint32_t> texnames;

        MaterialSet() {
			submats = std::vector<Material>(0); // init
            textures = std::vector<uint32_t>(0);
            textures.push_back(-1);

            freedomTextures = std::vector<uint32_t>(0);
            texnames = std::map<std::string, uint32_t>();
            init();
        }


        void clearSubmats() {
            submats.resize(0);
        }

        size_t addSubmat(const Material * submat) {
            size_t idx = submats.size();
            submats.push_back(*submat);
            return idx;
        }

        size_t addSubmat(const Material &submat) {
            return this->addSubmat(&submat);
        }

        void setSumbat(const size_t& i, const Material &submat) {
            if (submats.size() <= i) submats.resize(i+1);
            submats[i] = submat;
        }

        void loadToVGA();
        void bindWithContext(GLuint & prog);

		void freeTextureByGL(const GLuint& idx);
        void freeTexture(const uint32_t& idx);

        void clearGlTextures();

        uint32_t loadTexture(std::string tex, bool force_write = false);
        uint32_t loadTexture(const GLuint & gltexture);

        GLuint getGLTexture(const uint32_t & idx);
        uint32_t getTexture(const GLuint & idx);
    };
}

#include "materialSet.inl"