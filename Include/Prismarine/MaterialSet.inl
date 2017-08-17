#include "MaterialSet.hpp"

namespace ppr {

    inline void MaterialSet::init(){
        glCreateBuffers(1, &mats);
    }

    inline void MaterialSet::loadToVGA() {
        glNamedBufferData(mats, strided<VirtualMaterial>(submats.size()), submats.data(), GL_STATIC_DRAW);
        if (texset) texset->loadToVGA(); // load texture descriptors to VGA
    }

    inline void MaterialSet::bindWithContext(GLuint & prog) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, mats);
        if (texset) texset->bindWithContext(prog); // use texture set
    }

}
