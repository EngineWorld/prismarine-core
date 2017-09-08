#include "MaterialSet.hpp"

namespace NSM {

    inline void MaterialSet::init(){
        submats = std::vector<VirtualMaterial>(0); // init
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

    inline size_t MaterialSet::addSubmat(const VirtualMaterial * submat) {
        size_t idx = submats.size();
        submats.push_back(*submat);
        return idx;
    }

    inline size_t MaterialSet::addSubmat(const VirtualMaterial &submat) {
        return this->addSubmat(&submat);
    }

    inline void MaterialSet::setSumbat(const size_t& i, const VirtualMaterial &submat) {
        if (submats.size() <= i) submats.resize(i+1);
        submats[i] = submat;
    }

}
