#include "TriangleHierarchy.hpp"

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

namespace NSM {

    inline TriangleHierarchy::~TriangleHierarchy() {
        glDeleteProgram(geometryLoaderProgramI16);
        glDeleteProgram(geometryLoaderProgram2);
        glDeleteProgram(buildProgramH);
        glDeleteProgram(aabbMakerProgramH);
        glDeleteProgram(refitProgramH);
        glDeleteProgram(resortProgramH);
        glDeleteProgram(minmaxProgram2);

        glDeleteBuffers(1, &mat_triangle_ssbo);
        glDeleteTextures(1, &vbo_vertex_textrue);
        glDeleteTextures(1, &vbo_normal_textrue);
        glDeleteTextures(1, &vbo_texcoords_textrue);
        glDeleteTextures(1, &vbo_modifiers_textrue);
        glDeleteSamplers(1, &vbo_sampler);

        glDeleteBuffers(1, &geometryBlockUniform);
        glDeleteBuffers(1, &aabbCounter);
        glDeleteBuffers(1, &leafBuffer);
        glDeleteBuffers(1, &bvhnodesBuffer);
        glDeleteBuffers(1, &mortonBuffer);
        glDeleteBuffers(1, &mortonBufferIndex);
        glDeleteBuffers(1, &bvhflagsBuffer);
        glDeleteBuffers(1, &lscounterTemp);
        glDeleteBuffers(1, &tcounter);
        glDeleteBuffers(1, &minmaxBufRef);
        glDeleteBuffers(1, &minmaxBuf);
    }


    inline void TriangleHierarchy::initShaders() {
        //initShaderComputeSPIRV("./shaders-spv/hlbvh/refit.comp.spv", refitProgramH);
        //initShaderComputeSPIRV("./shaders-spv/hlbvh/build.comp.spv", buildProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/refit-new.comp.spv", refitProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/build-new.comp.spv", buildProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/aabbmaker.comp.spv", aabbMakerProgramH);
        initShaderComputeSPIRV("./shaders-spv/hlbvh/minmax.comp.spv", minmaxProgram2);
        initShaderComputeSPIRV("./shaders-spv/vertex/loader.comp.spv", geometryLoaderProgram2);
        initShaderComputeSPIRV("./shaders-spv/vertex/loader-int16.comp.spv", geometryLoaderProgramI16);
    }

    inline void TriangleHierarchy::init() {
        initShaders();
        sorter = new RadixSort();

        minmaxBufRef = allocateBuffer<bbox>(boundWorkSize);
        minmaxBuf = allocateBuffer<bbox>(boundWorkSize);
        lscounterTemp = allocateBuffer<uint32_t>(1);
        tcounter = allocateBuffer<uint32_t>(1);
        geometryBlockUniform = allocateBuffer<GeometryBlockUniform>(1);
        buildBuffer = allocateBuffer<uint32_t>(8);

        bbox bound;
        bound.mn.x = 100000.f;
        bound.mn.y = 100000.f;
        bound.mn.z = 100000.f;
        bound.mn.w = 100000.f;
        bound.mx.x = -100000.f;
        bound.mx.y = -100000.f;
        bound.mx.z = -100000.f;
        bound.mx.w = -100000.f;

        glNamedBufferSubData(lscounterTemp, 0, strided<int32_t>(1), zero);
        for (int i = 0; i < boundWorkSize;i++) {
            glNamedBufferSubData(minmaxBuf, strided<bbox>(i), strided<bbox>(1), &bound);
            glNamedBufferSubData(minmaxBufRef, strided<bbox>(i), strided<bbox>(1), &bound);
        }
    }

    inline void TriangleHierarchy::allocate(const size_t &count) {
        maxt = count * 2;

        size_t height = std::min(maxt > 0 ? (maxt - 1) / 2047 + 1 : 0, 2047u) + 1;

        vbo_vertex_textrue = allocateTexture2D<GL_RGBA32F>(6144, height);
        vbo_normal_textrue = allocateTexture2D<GL_RGBA32F>(6144, height);
        vbo_texcoords_textrue = allocateTexture2D<GL_RGBA32F>(6144, height);
        vbo_modifiers_textrue = allocateTexture2D<GL_RGBA32F>(6144, height);

        vbo_vertex_textrue_upload = allocateTexture2D<GL_RGBA32F>(6144, height);
        vbo_normal_textrue_upload = allocateTexture2D<GL_RGBA32F>(6144, height);
        vbo_texcoords_textrue_upload = allocateTexture2D<GL_RGBA32F>(6144, height);
        vbo_modifiers_textrue_upload = allocateTexture2D<GL_RGBA32F>(6144, height);

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
        activeBuffer = allocateBuffer<uint32_t>(maxt * boundWorkSize);
        mortonBuffer = allocateBuffer<uint64_t>(maxt * 1);
        mortonBufferIndex = allocateBuffer<uint32_t>(maxt * 1);
        leafBuffer = allocateBuffer<HlbvhNode>(maxt * 1);
        childBuffer = allocateBuffer<HlbvhNode>(maxt * 1);

        clearTribuffer();
    }

    inline void TriangleHierarchy::syncUniforms() {
        geometryBlockData.geometryUniform = geometryUniformData;
        glNamedBufferSubData(geometryBlockUniform, 0, strided<GeometryBlockUniform>(1), &geometryBlockData);

        this->bindUniforms();
    }

    inline void TriangleHierarchy::setMaterialID(int32_t id) {
        materialID = id;
    }

    inline void TriangleHierarchy::bindUniforms() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, geometryBlockUniform);
    }

    inline void TriangleHierarchy::bind() {
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

    inline void TriangleHierarchy::bindLeafs() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, leafBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, mortonBufferIndex);
    }

    inline void TriangleHierarchy::bindBVH() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, bvhnodesBuffer);
    }

    inline void TriangleHierarchy::clearTribuffer() {
        markDirty();
        glCopyNamedBufferSubData(minmaxBufRef, minmaxBuf, 0, 0, strided<bbox>(1));
        glCopyNamedBufferSubData(lscounterTemp, tcounter, 0, 0, strided<uint32_t>(1));
        geometryUniformData.triangleOffset = 0;
    }

    inline void TriangleHierarchy::configureIntersection(bool clearDepth) {
        this->geometryUniformData.clearDepth = clearDepth;
        this->syncUniforms();
    }

    inline void TriangleHierarchy::loadMesh(TriangleArrayInstance * gobject) {
        if (!gobject || gobject->meshUniformData.nodeCount <= 0) return;

        glCopyNamedBufferSubData(tcounter, gobject->meshUniformBuffer, 0, offsetof(MeshUniformStruct, storingOffset), sizeof(uint32_t));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tcounter);

        this->bind();
        gobject->bind();
        if (gobject->accessorSet) gobject->accessorSet->bind();
        if (gobject->bufferViewSet) gobject->bufferViewSet->bind();

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, mat_triangle_ssbo_upload);

        dispatchIndirect(gobject->index16bit ? geometryLoaderProgramI16 : geometryLoaderProgram2, gobject->indirect_dispatch_buffer);
        markDirty();

#ifdef PROFILE_RT
        glFinish();
#endif
    }

    inline bool TriangleHierarchy::isDirty() const {
        return dirty;
    }

    inline void TriangleHierarchy::markDirty() {
        dirty = true;
    }

    inline void TriangleHierarchy::resolve() {
        dirty = false;
    }

    inline void TriangleHierarchy::build(const glm::dmat4 &optimization) {
        //glFinish();
        // get triangle count that uploaded
        glGetNamedBufferSubData(tcounter, 0, strided<uint32_t>(1), &this->triangleCount);
        size_t triangleCount = std::min(uint32_t(this->triangleCount), uint32_t(maxt));
        geometryUniformData.triangleCount = triangleCount;

        // validate
        if (this->triangleCount <= 0 || !dirty) return;

        // copy uploading buffers to BVH
        size_t maxHeight = std::min(maxt > 0 ? (maxt - 1) / 2047 + 1 : 0, 2047u) + 1;
        size_t height = std::min(triangleCount > 0 ? (triangleCount - 1) / 2047 + 1 : 0, maxHeight) + 1;
        glCopyImageSubData(vbo_vertex_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_vertex_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 6144, height, 1);
        glCopyImageSubData(vbo_normal_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_normal_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 6144, height, 1);
        glCopyImageSubData(vbo_texcoords_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_texcoords_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 6144, height, 1);
        glCopyImageSubData(vbo_modifiers_textrue_upload, GL_TEXTURE_2D, 0, 0, 0, 0, vbo_modifiers_textrue, GL_TEXTURE_2D, 0, 0, 0, 0, 6144, height, 1);
        glCopyNamedBufferSubData(mat_triangle_ssbo_upload, mat_triangle_ssbo, 0, 0, std::min(uint32_t(this->triangleCount), uint32_t(maxt)) * sizeof(uint32_t));

        // use optimization matrix
        glCopyNamedBufferSubData(minmaxBufRef, minmaxBuf, 0, 0, strided<bbox>(1));
        {
            glm::dmat4 mat(1.0);
            mat *= glm::inverse(optimization);
            geometryUniformData.transform = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::mat4(mat)));
            geometryUniformData.transformInv = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(mat))));
        }



        // get bounding box of scene
        this->syncUniforms();
        this->bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, minmaxBuf);
        //dispatch(minmaxProgram2, 1);


        //glFinish();
        dispatch(minmaxProgram2, boundWorkSize);
        //glFinish();

        // getting boundings
        bbox * bounds = new bbox[boundWorkSize];
        glGetNamedBufferSubData(minmaxBuf, 0, strided<bbox>(boundWorkSize), bounds);
        bbox bound = bounds[0];
        for (int i = 1; i < boundWorkSize;i++) {
            bound.mn = glm::min(bounds[i].mn, bound.mn);
            bound.mx = glm::max(bounds[i].mx, bound.mx);
        }
        delete bounds;

        scale = (glm::make_vec4((float *)&bound.mx) - glm::make_vec4((float *)&bound.mn)).xyz();
        offset = glm::make_vec4((float *)&bound.mn).xyz();

        // fit BVH to boundings
        {
            glm::dmat4 mat(1.0);
            mat *= glm::inverse(glm::translate(glm::dvec3(offset)) * glm::scale(glm::dvec3(scale)));
            mat *= glm::inverse(glm::dmat4(optimization));
            geometryUniformData.transform = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::mat4(mat)));
            geometryUniformData.transformInv = *(Vc4x4 *)glm::value_ptr(glm::transpose(glm::inverse(glm::mat4(mat))));
        }

        // bind buffers
        glCopyNamedBufferSubData(lscounterTemp, aabbCounter, 0, 0, strided<uint32_t>(1));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mortonBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mortonBufferIndex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, aabbCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, leafBuffer);

        // get nodes morton codes and boundings
        this->bind();
        this->syncUniforms();
        dispatch(aabbMakerProgramH, tiled(triangleCount, worksize));
        glGetNamedBufferSubData(aabbCounter, 0, strided<uint32_t>(1), &triangleCount);
        //glFinish();
        if (triangleCount <= 0) return;
        geometryUniformData.triangleCount = triangleCount;

        // radix sort of morton-codes
        sorter->sort(mortonBuffer, mortonBufferIndex, triangleCount); // early serial tests
        //glFinish();

        // debug
        //std::vector<GLuint64> mortons(triangleCount);
        //glGetNamedBufferSubData(mortonBuffer, 0, strided<GLuint64>(mortons.size()), mortons.data());

        // bind BVH buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bvhnodesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvhflagsBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, activeBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, childBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, buildBuffer);

        for (int i=0;i<8;i++) glCopyNamedBufferSubData(lscounterTemp, buildBuffer, 0, i * sizeof(GLint), strided<GLint>(1));

        this->syncUniforms();
        GLint buildCounterData[8] = {0};
        for (int i = 0; i < 256; i++) {
            if ((i & 0xF) == 0xF) { // every 16 pass
                glGetNamedBufferSubData(buildBuffer, 0, 8 * sizeof(GLint), buildCounterData);
                GLint nodeCount = buildCounterData[5] - buildCounterData[4];
                if (nodeCount <= 0) break;
            }

            dispatch(buildProgramH, 32);
            glCopyNamedBufferSubData(buildBuffer, buildBuffer, 5 * sizeof(GLint), 4 * sizeof(GLint), sizeof(GLint));
            glCopyNamedBufferSubData(buildBuffer, buildBuffer, 2 * sizeof(GLint), 5 * sizeof(GLint), sizeof(GLint));
        }
        //glFinish();

        dispatch(refitProgramH, 32);
        //glFinish();

        //std::vector < HlbvhNode > nodes(triangleCount*2);
        //glGetNamedBufferSubData(bvhnodesBuffer, 0, strided<HlbvhNode>(nodes.size()), nodes.data());

        // set back triangle count
        this->geometryUniformData.triangleCount = this->triangleCount;
        this->syncUniforms();
        this->resolve();

        //glFinish();
    }
}
