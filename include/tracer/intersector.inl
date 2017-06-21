#include "intersector.hpp"

namespace Paper {

    inline void Intersector::initShaderCompute(std::string path, GLuint & prog) {
        std::string str = readSource(path);

        GLuint comp = glCreateShader(GL_COMPUTE_SHADER);
        {
            const char * strc = str.c_str();
            int32_t size = str.size();
            glShaderSource(comp, 1, &strc, &size);
            glCompileShader(comp);

            GLint status = false;
            glGetShaderiv(comp, GL_COMPILE_STATUS, &status);
            if (!status) {
                char * log = new char[1024];
                GLsizei len = 0;

                glGetShaderInfoLog(comp, 1024, &len, log);
                std::string logStr = std::string(log, len);
                std::cerr << logStr << std::endl;
            }
        }

        prog = glCreateProgram();
        glAttachShader(prog, comp);
        glLinkProgram(prog);

        GLint status = false;
        glGetProgramiv(prog, GL_LINK_STATUS, &status);
        if (!status) {
            char * log = new char[1024];
            GLsizei len = 0;

            glGetProgramInfoLog(prog, 1024, &len, log);
            std::string logStr = std::string(log, len);
            std::cerr << logStr << std::endl;
        }
    }

    inline void Intersector::initShaders() {
        initShaderCompute("./shaders/hlbvh/resort.comp", resortProgramH);
        initShaderCompute("./shaders/hlbvh/refit.comp", refitProgramH);
        initShaderCompute("./shaders/hlbvh/build.comp", buildProgramH);
        initShaderCompute("./shaders/hlbvh/aabbmaker.comp", aabbMakerProgramH);
        initShaderCompute("./shaders/hlbvh/minmax.comp", minmaxProgram2);
        initShaderCompute("./shaders/tools/loader.comp", geometryLoaderProgram2);
    }

    inline void Intersector::init() {
        initShaders();
        sorter = new RadixSort();

        glCreateBuffers(1, &minmaxBufRef);
        glNamedBufferStorage(minmaxBufRef, strided<bbox>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &minmaxBuf);
        glNamedBufferStorage(minmaxBuf, strided<bbox>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &lscounterTemp);
        glNamedBufferStorage(lscounterTemp, strided<uint32_t>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &tcounter);
        glNamedBufferStorage(tcounter, strided<uint32_t>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &geometryBlockUniform);
        glNamedBufferStorage(geometryBlockUniform, strided<GeometryBlockUniform>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        bbox bound;
        bound.mn.x = 100000.f;
        bound.mn.y = 100000.f;
        bound.mn.z = 100000.f;
        bound.mn.w = 100000.f;
        bound.mx.x = -100000.f;
        bound.mx.y = -100000.f;
        bound.mx.z = -100000.f;
        bound.mx.w = -100000.f;

        glNamedBufferSubData(minmaxBuf, 0, strided<bbox>(1), &bound);
        glNamedBufferSubData(minmaxBufRef, 0, strided<bbox>(1), &bound);
    }

    inline void Intersector::allocate(const size_t &count) {
        maxt = count;


        glCreateBuffers(1, &ebo_triangle_ssbo);
        glNamedBufferStorage(ebo_triangle_ssbo, strided<int32_t>(maxt * 3), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &vbo_triangle_ssbo);
        glNamedBufferStorage(vbo_triangle_ssbo, strided<VboDataStride>(maxt * 3), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &mat_triangle_ssbo);
        glNamedBufferStorage(mat_triangle_ssbo, strided<int32_t>(maxt), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &nodeCounter);
        glNamedBufferStorage(nodeCounter, strided<int32_t>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &aabbCounter);
        glNamedBufferStorage(aabbCounter, strided<int32_t>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &numBuffer);
        glNamedBufferStorage(numBuffer, strided<glm::ivec2>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);



        glCreateBuffers(1, &mortonBuffer);
        glNamedBufferStorage(mortonBuffer, strided<int32_t>(maxt), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &mortonBufferIndex);
        glNamedBufferStorage(mortonBufferIndex, strided<int32_t>(maxt), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &bvhnodesBuffer);
        glNamedBufferStorage(bvhnodesBuffer, strided<HlbvhNode>(maxt * 2), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &bvhflagsBuffer);
        glNamedBufferStorage(bvhflagsBuffer, strided<uint32_t>(maxt * 2), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &leafBufferSorted);
        glNamedBufferStorage(leafBufferSorted, strided<Leaf>(maxt), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &leafBuffer);
        glNamedBufferStorage(leafBuffer, strided<Leaf>(maxt), nullptr, GL_DYNAMIC_STORAGE_BIT);

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
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, vbo_triangle_ssbo);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ebo_triangle_ssbo);
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
        geometryUniformData.transform = *(Vc4x4 *)glm::value_ptr(gobject->trans);
        geometryUniformData.transformInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(gobject->trans));
        geometryUniformData.texmatrix = *(Vc4x4 *)glm::value_ptr(gobject->texmat);
        geometryUniformData.colormod = *(Vc4 *)glm::value_ptr(gobject->colormod);
        geometryUniformData.offset = gobject->voffset;
        attributeUniformData = gobject->attributeUniformData;

        gobject->bind();
        this->bind();
        this->syncUniforms();

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, tcounter);

        glUseProgram(geometryLoaderProgram2);
        glDispatchCompute(tiled(gobject->nodeCount, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        markDirty();

        glGetNamedBufferSubData(tcounter, 0, strided<uint32_t>(1), &triangleCount);
        verticeCount = triangleCount * 3;
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
            octreeUniformData.project = *(Vc4x4 *)glm::value_ptr(glm::mat4(mat));
            octreeUniformData.unproject = *(Vc4x4 *)glm::value_ptr(glm::inverse(glm::mat4(mat)));
        }

        this->syncUniforms();
        this->bind();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, minmaxBuf);
        glUseProgram(minmaxProgram2);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        bbox bound;
        glGetNamedBufferSubData(minmaxBuf, 0, strided<bbox>(1), &bound);
        scale = (glm::make_vec4((float *)&bound.mx) - glm::make_vec4((float *)&bound.mn)).xyz();
        offset = glm::make_vec4((float *)&bound.mn).xyz();

        {
            glm::dmat4 mat(1.0f);
            mat = glm::scale(mat, 1.0 / glm::dvec3(scale));
            mat = glm::translate(mat, -glm::dvec3(offset));
            mat *= glm::inverse(glm::dmat4(optimization));
            octreeUniformData.project = *(Vc4x4 *)glm::value_ptr(glm::mat4(mat));
            octreeUniformData.unproject = *(Vc4x4 *)glm::value_ptr(glm::inverse(glm::mat4(mat)));
        }

        glCopyNamedBufferSubData(lscounterTemp, aabbCounter, 0, 0, strided<uint32_t>(1));
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, aabbCounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, leafBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mortonBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mortonBufferIndex);

        this->bind();
        this->syncUniforms();
        glUseProgram(aabbMakerProgramH);
        glDispatchCompute(tiled(triangleCount, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        glGetNamedBufferSubData(aabbCounter, 0, strided<uint32_t>(1), &triangleCount);
        if (triangleCount <= 0) return;

        // radix sort of morton-codes
        sorter->sort(mortonBuffer, mortonBufferIndex, triangleCount); // early serial tests
        geometryUniformData.triangleCount = triangleCount;


        //std::vector<GLuint> radixTest = std::vector<GLuint>(triangleCount);
        //glGetNamedBufferSubData(mortonBuffer, 0, strided<GLuint>(triangleCount), radixTest.data());


        this->syncUniforms();
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, leafBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mortonBufferIndex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, leafBufferSorted);
        glUseProgram(resortProgramH);
        glDispatchCompute(tiled(triangleCount, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, numBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mortonBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, mortonBufferIndex);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, leafBufferSorted);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, bvhnodesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, bvhflagsBuffer);
        glCopyNamedBufferSubData(lscounterTemp, nodeCounter, 0, 0, strided<uint32_t>(1));

        glm::ivec2 range = glm::ivec2(0, triangleCount);
        glNamedBufferSubData(numBuffer, 0, strided<glm::ivec2>(1), &range);

        glUseProgram(buildProgramH);
        glDispatchCompute(tiled(triangleCount, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        glUseProgram(refitProgramH);
        glDispatchCompute(tiled(triangleCount, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }
}
