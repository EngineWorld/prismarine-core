#include "MaterialSet.hpp"

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

namespace NSM {

    inline void MaterialSet::init(){
        submats = std::vector<VirtualMaterial>(0); // init
        glCreateBuffers(1, &mats);
		glCreateBuffers(1, &countBuffer);
    }

    inline void MaterialSet::loadToVGA() {
		GLint offsetSize[2] = { loadOffset, submats.size() };
		glNamedBufferData(countBuffer, sizeof(GLint)*2, offsetSize, GL_STATIC_DRAW);
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

	inline size_t MaterialSet::getMaterialCount() {
		return submats.size();
	}

	inline void MaterialSet::setLoadingOffset(GLint loadOffset) {
		this->loadOffset = loadOffset;
	}

}
