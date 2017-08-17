#pragma once

#include "Utils.hpp"
#include "Structs.hpp"

namespace ppr {
    class TextureSet : public BaseClass {

    private:
        GLuint firstBind = 6;
        GLuint texturesBuffer = -1;
        //std::vector<uint64_t> vctr;
        std::vector<GLint> vctr;
        void init();

    public:

        std::vector<uint32_t> textures;
        std::vector<uint32_t> freedomTextures;
        std::map<std::string, uint32_t> texnames;

        TextureSet() {
            textures = std::vector<uint32_t>(0);
            textures.push_back(-1);
            freedomTextures = std::vector<uint32_t>(0);
            texnames = std::map<std::string, uint32_t>();
            init();
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

#include "TextureSet.inl"