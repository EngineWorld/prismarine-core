#include "material.hpp"
#include <algorithm>
#include <numeric>
#include <list>

namespace Paper {

    inline void Material::init(){
        glCreateBuffers(1, &mats);
    }

    inline void Material::loadToVGA() {
        glNamedBufferData(mats, strided<Submat>(submats.size()), submats.data(), GL_STATIC_DRAW);
    }

    inline void Material::bindWithContext(GLuint & prog) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, mats);

        const GLuint firstBind = 0;
        const GLuint textureLocation = 1;
        uint32_t pcount = std::min((uint32_t)samplers.size(), 128u);

        std::vector<uint64_t> vctr(pcount);
        for (int i = 0; i < pcount; i++) {
            uint64_t texHandle = glGetTextureHandleARB(samplers[i]);
            vctr[i] = texHandle;
        }
        glProgramUniformHandleui64vARB(prog, textureLocation, pcount, vctr.data());

        //std::vector<uint32_t> vctr(pcount);
        //for (int i = 0; i < pcount; i++) vctr[i] = firstBind + i;
        //glBindTextures(firstBind, pcount, samplers.data());
        //glProgramUniform1iv(prog, textureLocation, pcount, vctr.data());
    }

    inline void Material::freeTexture(const uint32_t& idx) {
        freedomSamplers.push_back(idx);
        samplers[idx] = -1;
    }

	inline void Material::freeTextureByGL(const GLuint & gltexture) {
        for (int i = 1; i < samplers.size(); i++) {
            if (samplers[i] == gltexture) {
                this->freeTexture(i);
            }
        }
	}

    // get texture by GL
    inline uint32_t Material::getTexture(const GLuint & gltexture) {
        for (int i = 1; i < samplers.size(); i++) {
            if (samplers[i] == gltexture) return i;
        }
        return 0;
    }

    inline GLuint Material::getGLTexture(const uint32_t & idx) {
        return samplers[idx];
    }

    inline uint32_t Material::loadTexture(const GLuint & gltexture) {
        int32_t idx = getTexture(gltexture);
        if (idx) return idx;
        if (freedomSamplers.size() > 0) {
            idx = freedomSamplers[freedomSamplers.size() - 1];
            freedomSamplers.pop_back();
            samplers[idx] = gltexture;
        }
        else {
            idx = samplers.size();
            samplers.push_back(gltexture);
        }
        return idx;
    };

#ifdef USE_FREEIMAGE
    inline uint32_t Material::loadTexture(std::string tex, bool force_write) {
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
        glTextureSubImage2D(texture, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixelsPtr);

        texnames[tex] = texture;
        return this->loadTexture(texture);
    }
#endif

}
