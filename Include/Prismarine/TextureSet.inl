#include "TextureSet.hpp"

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

namespace NSM {

    inline void TextureSet::init() {
        textures = std::vector<uint32_t>(0);
        textures.push_back(-1);
        freedomTextures = std::vector<uint32_t>(0);
        texnames = std::map<std::string, uint32_t>();
        glCreateBuffers(1, &texturesBuffer);
    }

    inline void TextureSet::loadToVGA() {
        
        uint32_t pcount = std::min((uint32_t)textures.size(), 64u);
        vctr.resize(pcount);
        for (int i = 0; i < pcount; i++) {
        uint64_t texHandle = glGetTextureHandleARB(textures[i]);
        glMakeTextureHandleResidentARB(texHandle);
        vctr[i] = texHandle;
        }
        glNamedBufferData(texturesBuffer, vctr.size() * sizeof(GLuint64), vctr.data(), GL_STATIC_DRAW);
        

        /*
        uint32_t pcount = std::min((uint32_t)textures.size(), 64u);
        vctr.resize(pcount);
        for (int i = 0; i < pcount; i++) vctr[i] = firstBind + i - 1; // use bindings, except first (move indice)
        */
    }

    inline void TextureSet::bindWithContext(GLuint & prog) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, texturesBuffer); // bindless texture buffer
        glProgramUniformHandleui64vARB(prog, 1, vctr.size(), vctr.data()); // bindless texture (uniform)

        //glBindTextures(firstBind, vctr.size() - 1, textures.data() + 1); // bind textures, except first
        //glProgramUniform1iv(prog, 1, vctr.size(), vctr.data());
    }

    inline void TextureSet::clearGlTextures() {
        for (int i = 1; i < textures.size(); i++) {
            this->freeTexture(i);
        }
    }

    inline void TextureSet::freeTexture(const uint32_t& idx) {
        freedomTextures.push_back(idx);
        textures[idx] = -1;
    }

    inline void TextureSet::freeTextureByGL(const GLuint & gltexture) {
        for (int i = 1; i < textures.size(); i++) {
            if (textures[i] == gltexture) {
                this->freeTexture(i);
            }
        }
    }

    // get texture by GL
    inline uint32_t TextureSet::getTexture(const GLuint & gltexture) {
        for (int i = 1; i < textures.size(); i++) {
            if (textures[i] == gltexture && textures[i] != -1) return i;
        }
        return 0;
    }

    inline GLuint TextureSet::getGLTexture(const uint32_t & idx) {
        return textures[idx];
    }

    inline uint32_t TextureSet::loadTexture(const GLuint & gltexture) {
        int32_t idx = getTexture(gltexture);
        if (idx && idx >= 0 && idx != -1) return idx;
        if (freedomTextures.size() > 0) {
            idx = freedomTextures[freedomTextures.size() - 1];
            freedomTextures.pop_back();
            textures[idx] = gltexture;
        }
        else {
            idx = textures.size();
            textures.push_back(gltexture);
        }
        return idx;
    };


#ifdef USE_FREEIMAGE
    inline uint32_t TextureSet::loadTexture(std::string tex, bool force_write) {
        if (tex == "") return 0;
        if (!force_write && texnames.find(tex) != texnames.end()) {
            return getTexture(texnames[tex]); // if already in dictionary
        }

        FREE_IMAGE_FORMAT formato = FreeImage_GetFileType(tex.c_str(), 0);
        if (formato == FIF_UNKNOWN) {
            return 0;
        }
        FIBITMAP* imagen = FreeImage_Load(formato, tex.c_str());
        if (!imagen) {
            return 0;
        }

        FIBITMAP* temp = FreeImage_ConvertTo32Bits(imagen);
        FreeImage_Unload(imagen);
        imagen = temp;

        uint32_t width = FreeImage_GetWidth(imagen);
        uint32_t height = FreeImage_GetHeight(imagen);
        uint8_t * pixelsPtr = FreeImage_GetBits(imagen);

        GLuint texture = 0;
        glCreateTextures(GL_TEXTURE_2D, 1, &texture);
        glTextureStorage2D(texture, 1, GL_RGBA8, width, height);
        glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureSubImage2D(texture, 0, 0, 0, width, height, GL_BGRA, GL_UNSIGNED_BYTE, pixelsPtr);

        texnames[tex] = texture;
        return this->loadTexture(texture);
    }
#endif

}
