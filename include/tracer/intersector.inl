#include "intersector.hpp"

namespace Paper {

    inline void Intersector::initShaders() {
        initShaderCompute("./shaders/hlbvh/refit.comp", refitProgramH);
        initShaderCompute("./shaders/hlbvh/build.comp", buildProgramH);
        initShaderCompute("./shaders/hlbvh/aabbmaker.comp", aabbMakerProgramH);
        initShaderCompute("./shaders/hlbvh/minmax.comp", minmaxProgram2);
        initShaderCompute("./shaders/tools/loader.comp", geometryLoaderProgram2);
        initShaderCompute("./shaders/tools/loader-int16.comp", geometryLoaderProgramI16);
        
        /*
        initShaderComputeSPIRV("./shaders-spv/hlbvh/refit.comp.spv", refitProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/build.comp.spv", buildProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/aabbmaker.comp.spv", aabbMakerProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/minmax.comp.spv", minmaxProgram2);
        initShaderComputeSPIRV("./shaders-spv/tools/loader.comp.spv", geometryLoaderProgram2);
        initShaderComputeSPIRV("./shaders-spv/tools/loader-int16.comp.spv", geometryLoaderProgramI16);
        */
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

        glCreateSamplers(1, &vbo_sampler);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(vbo_sampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        mat_triangle_ssbo = allocateBuffer<int32_t>(maxt);
        aabbCounter = allocateBuffer<int32_t>(1);
        bvhnodesBuffer = allocateBuffer<HlbvhNode>(maxt * 2);
        bvhflagsBuffer = allocateBuffer<uint32_t>(maxt * 2);
        mortonBuffer = allocateBuffer<uint32_t>(maxt * 1);
        mortonBufferIndex = allocateBuffer<uint32_t>(maxt * 1);
        leafBuffer = allocateBuffer<Leaf>(maxt * 1);

        clearTribuffer();
    }

    inline void Intersector::syncUniforms() {
        geometryBlockData.geometryUniform = geometryUniformData;
        geometryBlockData.attributeUniform = attributeUniformData;
        geometryBlockData.octreeUniform = octreeUniformData;
        geometryBlockData.minmaxUniform = minmaxUniformData;
        glNamedBufferSubData(geometryBlockUniform, 0, strided<GeometryBlockUniform>(1), &geometryBlockData);

        this->bindUniforms();
    }

    inline void Intersector::setMaterialID(int32_t id) {
        materialID = id;
    }

    inline void Intersector::bindUniforms() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, geometryBlockUniform);
    }

    inline void Intersector::bind() {
        // mosaic buffers for write
        glBindImageTexture(0, vbo_vertex_textrue, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(1, vbo_normal_textrue, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(2, vbo_texcoords_textrue, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(3, vbo_modifiers_textrue, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

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

        // legacy SSBO
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, mat_triangle_ssbo);
    }

    inline void Intersector::bindBVH() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, bvhnodesBuffer);
    }

    inline void Intersector::clearTribuffer() {
        triangleCount = 0;
        verticeCount = 0;
        markDirty();

        glCopyNamedBufferSubData(minmaxBufRef, minmaxBuf, 0, 0, strided<bbox>(1));
        glCopyNamedBufferSubData(lscounterTemp, tcounter, 0, 0, strided<uint32_t>(1));
    }

    inline void Intersector::configureIntersection(bool clearDepth) {
        this->geometryUniformData.triangleCount = this->triangleCount;
        this->geometryUniformData.clearDepth = clearDepth;
        this->syncUniforms();
    }

    inline void Intersector::loadMesh(Mesh * gobject) {
        if (!gobject || gobject->nodeCount  <= 0) return;

        geometryUniformData.unindexed = gobject->unindexed;
        geometryUniformData.loadOffset = gobject->offset;
        geometryUniformData.materialID = gobject->materialID;
        geometryUniformData.triangleOffset = triangleCount;
        geometryUniformData.triangleCount = gobject->nodeCount;
        geometryUniformData.transform = *(Vc4x4 *)glm::value_ptr(glm::transpose(gobject->trans));
        geometryUniformData.transformInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(gobject->trans));
        geometryUniformData.texmatrix = *(Vc4x4 *)glm::value_ptr(gobject->texmat);
        geometryUniformData.colormod = *(Vc4 *)glm::value_ptr(gobject->colormod);
        geometryUniformData.offset = gobject->voffset;
        attributeUniformData = gobject->attributeUniformData;

        gobject->bind();
        this->bind();
        this->syncUniforms();

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tcounter);

        dispatch(gobject->index16bit ? geometryLoaderProgramI16 : geometryLoaderProgram2, tiled(gobject->nodeCount, worksize));
        markDirty();

        glGetNamedBufferSubData(tcounter, 0, strided<uint32_t>(1), &triangleCount);
        verticeCount = triangleCount * 3;

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
        if (this->triangleCount <= 0 || !dirty) return;
        this->resolve();

        size_t triangleCount = this->triangleCount;
        const double prec = 1000000.0;
        
        glCopyNamedBufferSubData(minmaxBufRef, minmaxBuf, 0, 0, strided<bbox>(1));
        geometryUniformData.triangleOffset = 0;
        geometryUniformData.triangleCount = triangleCount;
        minmaxUniformData.prec = prec;
        minmaxUniformData.heap = 1;

        {
            glm::dmat4 mat(1.0);
            mat *= glm::inverse(optimization);
            octreeUniformData.project = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::mat4(mat)));
            octreeUniformData.unproject = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(mat))));
        }

        this->syncUniforms();
        this->bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, minmaxBuf);
        dispatch(minmaxProgram2, 1);

        glGetNamedBufferSubData(minmaxBuf, 0, strided<bbox>(1), &bound);
        scale = (glm::make_vec4((float *)&bound.mx) - glm::make_vec4((float *)&bound.mn)).xyz();
        offset = glm::make_vec4((float *)&bound.mn).xyz();

        {
            glm::dmat4 mat(1.0);
            mat = glm::scale(mat, 1.0 / glm::dvec3(scale));
            mat = glm::translate(mat, -glm::dvec3(offset));
            mat *= glm::inverse(glm::dmat4(optimization));

            octreeUniformData.project = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::mat4(mat)));
            octreeUniformData.unproject = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(mat))));
        }

        glCopyNamedBufferSubData(lscounterTemp, aabbCounter, 0, 0, strided<uint32_t>(1));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, leafBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mortonBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mortonBufferIndex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, aabbCounter);

        this->bind();
        this->syncUniforms();
        dispatch(aabbMakerProgramH, tiled(triangleCount, worksize));

        glGetNamedBufferSubData(aabbCounter, 0, strided<uint32_t>(1), &triangleCount);
        if (triangleCount <= 0) return;

        // radix sort of morton-codes
        sorter->sort(mortonBuffer, mortonBufferIndex, triangleCount); // early serial tests
        geometryUniformData.triangleCount = triangleCount;

        //std::vector<GLuint> mortons(triangleCount);
        //glGetNamedBufferSubData(mortonBufferIndex, 0, strided<GLuint>(mortons.size()), mortons.data());

        this->syncUniforms();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, leafBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mortonBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mortonBufferIndex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, aabbCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bvhnodesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvhflagsBuffer);

        dispatch(buildProgramH, 1);
        dispatch(refitProgramH, tiled(triangleCount, worksize));
    }
}
