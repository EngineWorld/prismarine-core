#include "intersector.hpp"

namespace Paper {

    inline void Intersector::initShaders() {
        /*
        initShaderCompute("./shaders/hlbvh/refit.comp", refitProgramH);
        initShaderCompute("./shaders/hlbvh/build.comp", buildProgramH);
        initShaderCompute("./shaders/hlbvh/aabbmaker.comp", aabbMakerProgramH);
        initShaderCompute("./shaders/hlbvh/minmax.comp", minmaxProgram2);
        initShaderCompute("./shaders/tools/loader.comp", geometryLoaderProgram2);
        initShaderCompute("./shaders/tools/loader-int16.comp", geometryLoaderProgramI16);
        */

        initShaderComputeSPIRV("./shaders-spv/hlbvh/refit.comp.spv", refitProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/build.comp.spv", buildProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/aabbmaker.comp.spv", aabbMakerProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/minmax.comp.spv", minmaxProgram2);
        initShaderComputeSPIRV("./shaders-spv/tools/loader.comp.spv", geometryLoaderProgram2);
        initShaderComputeSPIRV("./shaders-spv/tools/loader-int16.comp.spv", geometryLoaderProgramI16);
    }

    inline void Intersector::init() {
        initShaders();
        sorter = new RadixSort();

        minmaxBufRef = allocateBuffer<bbox>(1);
        minmaxBuf = allocateBuffer<bbox>(1);
        lscounterTemp = allocateBuffer<uint32_t>(1);
        tcounter = allocateBuffer<uint32_t>(1);
        geometryBlockUniform = allocateBuffer<GeometryBlockUniform>(1);

        bound.mn.x = 100000.f;
        bound.mn.y = 100000.f;
        bound.mn.z = 100000.f;
        bound.mn.w = 100000.f;
        bound.mx.x = -100000.f;
        bound.mx.y = -100000.f;
        bound.mx.z = -100000.f;
        bound.mx.w = -100000.f;

        glNamedBufferSubData(lscounterTemp, 0, strided<int32_t>(1), zero);
        glNamedBufferSubData(minmaxBuf, 0, strided<bbox>(1), &bound);
        glNamedBufferSubData(minmaxBufRef, 0, strided<bbox>(1), &bound);
    }

    inline void Intersector::allocate(const size_t &count) {
        maxt = count;

        vbo_vertex_textrue = allocateTexture2D<GL_RGBA32F>(3072, 1024);
        vbo_normal_textrue = allocateTexture2D<GL_RGBA32F>(3072, 1024);
        vbo_texcoords_textrue = allocateTexture2D<GL_RGBA32F>(3072, 1024);
        vbo_modifiers_textrue = allocateTexture2D<GL_RGBA32F>(3072, 1024);

        vbo_vertex_textrue_upload = allocateTexture2D<GL_RGBA32F>(3072, 1024);
        vbo_normal_textrue_upload = allocateTexture2D<GL_RGBA32F>(3072, 1024);
        vbo_texcoords_textrue_upload = allocateTexture2D<GL_RGBA32F>(3072, 1024);
        vbo_modifiers_textrue_upload = allocateTexture2D<GL_RGBA32F>(3072, 1024);

        glCreateSamplers(1, &vbo_sampler);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        mat_triangle_ssbo = allocateBuffer<int32_t>(maxt);
        mat_triangle_ssbo_upload = allocateBuffer<int32_t>(maxt);

        aabbCounter = allocateBuffer<int32_t>(1);
        bvhnodesBuffer = allocateBuffer<HlbvhNode>(maxt * 2);
        bvhflagsBuffer = allocateBuffer<uint32_t>(maxt * 2);
        mortonBuffer = allocateBuffer<uint32_t>(maxt * 1);
        mortonBufferIndex = allocateBuffer<uint32_t>(maxt * 1);
        leafBuffer = allocateBuffer<HlbvhNode>(maxt * 1);
        activeBuffer = allocateBuffer<uint32_t>(maxt * 2);

        clearTribuffer();
    }

    inline void Intersector::syncUniforms() {
        geometryBlockData.geometryUniform = geometryUniformData;
        glNamedBufferSubData(geometryBlockUniform, 0, strided<GeometryBlockUniform>(1), &geometryBlockData);

        this->bindUniforms();
    }

    inline void Intersector::setMaterialID(int32_t id) {
        materialID = id;
    }

    inline void Intersector::bindUniforms() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, geometryBlockUniform);
    }

    inline void Intersector::bind() {
        // mosaic buffers for write
        glBindImageTexture(0, vbo_vertex_textrue_upload, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(1, vbo_normal_textrue_upload, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(2, vbo_texcoords_textrue_upload, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(3, vbo_modifiers_textrue_upload, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

        // use mosaic sampler
        glBindSampler(0, vbo_sampler);
        glBindSampler(1, vbo_sampler);
        glBindSampler(2, vbo_sampler);
        glBindSampler(3, vbo_sampler);

        // mosaic buffer
        glBindTextureUnit(0, vbo_vertex_textrue);
        glBindTextureUnit(1, vbo_normal_textrue);
        glBindTextureUnit(2, vbo_texcoords_textrue);
        glBindTextureUnit(3, vbo_modifiers_textrue);

        // material SSBO
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, mat_triangle_ssbo);
    }

    inline void Intersector::bindBVH() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, bvhnodesBuffer);
    }

    inline void Intersector::clearTribuffer() {
        markDirty();
        glCopyNamedBufferSubData(minmaxBufRef, minmaxBuf, 0, 0, strided<bbox>(1));
        glCopyNamedBufferSubData(lscounterTemp, tcounter, 0, 0, strided<uint32_t>(1));
        geometryUniformData.triangleOffset = 0;
    }

    inline void Intersector::configureIntersection(bool clearDepth) {
        this->geometryUniformData.clearDepth = clearDepth;
        this->syncUniforms();
    }

    inline void Intersector::loadMesh(Mesh * gobject) {
        if (!gobject || gobject->meshUniformData.nodeCount <= 0) return;

        glCopyNamedBufferSubData(tcounter, gobject->meshUniformBuffer, 0, offsetof(MeshUniformStruct, storingOffset), sizeof(uint32_t));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tcounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mat_triangle_ssbo_upload);

        this->bind();
        gobject->bind();

        dispatchIndirect(gobject->index16bit ? geometryLoaderProgramI16 : geometryLoaderProgram2, gobject->indirect_dispatch_buffer);
        markDirty();

        glUseProgram(0);
    }

    inline bool Intersector::isDirty() const {
        return dirty;
    }

    inline void Intersector::markDirty() {
        dirty = true;
    }

    inline void Intersector::resolve() {
        dirty = false;
    }

    inline void Intersector::build(const glm::dmat4 &optimization) {
        // get triangle count that uploaded
        glGetNamedBufferSubData(tcounter, 0, strided<uint32_t>(1), &this->triangleCount);
        geometryUniformData.triangleCount = triangleCount;

        // validate
        if (this->triangleCount <= 0 || !dirty) return;

        // copy uploading buffers to BVH
        glCopyImageSubData(vbo_vertex_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_vertex_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 3072, (this->triangleCount > 0 ? (this->triangleCount - 1) / 1023 + 1 : 0) + 1, 1);
        glCopyImageSubData(vbo_normal_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_normal_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 3072, (this->triangleCount > 0 ? (this->triangleCount - 1) / 1023 + 1 : 0) + 1, 1);
        glCopyImageSubData(vbo_texcoords_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_texcoords_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 3072, (this->triangleCount > 0 ? (this->triangleCount - 1) / 1023 + 1 : 0) + 1, 1);
        glCopyImageSubData(vbo_modifiers_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_modifiers_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 3072, (this->triangleCount > 0 ? (this->triangleCount - 1) / 1023 + 1 : 0) + 1, 1);
        glCopyNamedBufferSubData(mat_triangle_ssbo_upload, mat_triangle_ssbo, 0, 0, this->triangleCount * sizeof(uint32_t));
        this->resolve();

        size_t triangleCount = this->triangleCount;
        const double prec = 1000000.0;
        
        glCopyNamedBufferSubData(minmaxBufRef, minmaxBuf, 0, 0, strided<bbox>(1));
        {
            glm::dmat4 mat(1.0);
            mat *= glm::inverse(optimization);
            geometryUniformData.transform = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::mat4(mat)));
            geometryUniformData.transformInv = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(mat))));
        }

        this->syncUniforms();
        this->bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, minmaxBuf);
        dispatch(minmaxProgram2, 1);

        glGetNamedBufferSubData(minmaxBuf, 0, strided<bbox>(1), &bound);
        scale = (glm::make_vec4((float *)&bound.mx) - glm::make_vec4((float *)&bound.mn)).xyz();
        offset = glm::make_vec4((float *)&bound.mn).xyz();

        {
            glm::dmat4 mat(1.0);
            mat = glm::scale(mat, 1.0 / glm::dvec3(scale));
            mat = glm::translate(mat, -glm::dvec3(offset));
            mat *= glm::inverse(glm::dmat4(optimization));

            geometryUniformData.transform = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::mat4(mat)));
            geometryUniformData.transformInv = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(mat))));
        }

        glCopyNamedBufferSubData(lscounterTemp, aabbCounter, 0, 0, strided<uint32_t>(1));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mortonBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mortonBufferIndex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, aabbCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, leafBuffer);

        this->bind();
        this->syncUniforms();
        dispatch(aabbMakerProgramH, tiled(triangleCount, worksize));

        glGetNamedBufferSubData(aabbCounter, 0, strided<uint32_t>(1), &triangleCount);
        if (triangleCount <= 0) return;

        // radix sort of morton-codes
        sorter->sort(mortonBuffer, mortonBufferIndex, triangleCount); // early serial tests
        geometryUniformData.triangleCount = triangleCount;

        //std::vector<GLuint> mortons(triangleCount);
        //glGetNamedBufferSubData(mortonBuffer, 0, strided<GLuint>(mortons.size()), mortons.data());

        this->syncUniforms();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bvhnodesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvhflagsBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, activeBuffer);

        dispatch(buildProgramH, 1);
        dispatch(refitProgramH, tiled(triangleCount, worksize));

        // set back triangle count
        this->geometryUniformData.triangleCount = this->triangleCount;
        this->syncUniforms();
    }
}
