#include "tracer.hpp"

namespace Paper {

    inline void Tracer::initShaders() {
        
        /*
#ifdef USE_OPTIMIZED_RT
        initShaderCompute("./shaders/render/testmat-rt.comp", matProgram);
#else
        initShaderCompute("./shaders/render/testmat.comp", matProgram);
#endif

        initShaderCompute("./shaders/render/begin.comp", beginProgram);
        initShaderCompute("./shaders/render/reclaim.comp", reclaimProgram);
        initShaderCompute("./shaders/render/camera.comp", cameraProgram);
        initShaderCompute("./shaders/render/clear.comp", clearProgram);
        initShaderCompute("./shaders/render/sampler.comp", samplerProgram);
        initShaderCompute("./shaders/render/intersection.comp", intersectionProgram);
        */

#ifdef USE_OPTIMIZED_RT
        initShaderComputeSPIRV("./shaders-spv/render/testmat-rt.comp.spv", matProgram);
#else
        initShaderComputeSPIRV("./shaders-spv/render/testmat.comp.spv", matProgram);
#endif

        initShaderComputeSPIRV("./shaders-spv/render/begin.comp.spv", beginProgram);
        initShaderComputeSPIRV("./shaders-spv/render/reclaim.comp.spv", reclaimProgram);
        initShaderComputeSPIRV("./shaders-spv/render/camera.comp.spv", cameraProgram);
        initShaderComputeSPIRV("./shaders-spv/render/clear.comp.spv", clearProgram);
        initShaderComputeSPIRV("./shaders-spv/render/sampler.comp.spv", samplerProgram);
        initShaderComputeSPIRV("./shaders-spv/render/intersection.comp.spv", intersectionProgram);


        {
            GLuint vert = glCreateShader(GL_VERTEX_SHADER);
            {
                std::string path = "./shaders/render/render.vert";
                std::string str = readSource(path);

                const char * strc = str.c_str();
                int32_t size = str.size();
                glShaderSource(vert, 1, &strc, &size);
                glCompileShader(vert);

                GLint status = false;
                glGetShaderiv(vert, GL_COMPILE_STATUS, &status);
                if (!status) {
                    char * log = new char[1024];
                    GLsizei len = 0;

                    glGetShaderInfoLog(vert, 1024, &len, log);
                    std::string logStr = std::string(log, len);
                    std::cerr << logStr << std::endl;
                }
            }

            GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
            {
                std::string path = "./shaders/render/render.frag";
                std::string str = readSource(path);
                const char * strc = str.c_str();
                int32_t size = str.size();
                glShaderSource(frag, 1, &strc, &size);
                glCompileShader(frag);

                GLint status = false;
                glGetShaderiv(frag, GL_COMPILE_STATUS, &status);
                if (!status) {
                    char * log = new char[1024];
                    GLsizei len = 0;

                    glGetShaderInfoLog(frag, 1024, &len, log);
                    std::string logStr = std::string(log, len);
                    std::cerr << logStr << std::endl;
                }
            }

            renderProgram = glCreateProgram();
            glAttachShader(renderProgram, vert);
            glAttachShader(renderProgram, frag);
            glLinkProgram(renderProgram);

            GLint status = false;
            glGetProgramiv(renderProgram, GL_LINK_STATUS, &status);
            if (!status) {
                char * log = new char[1024];
                GLsizei len = 0;

                glGetProgramInfoLog(renderProgram, 1024, &len, log);
                std::string logStr = std::string(log, len);
                std::cerr << logStr << std::endl;
            }
        }
    }

    inline void Tracer::init() {
        initShaders();
        lightUniformData = new LightUniformStruct[6];
        sorter = new RadixSort();

        for (int i = 0; i < 6;i++) {
            lightColor[i] = glm::vec4((glm::vec3(255.f, 250.f, 244.f) / 255.f) * 100.f, 40.0f);
            lightAmbient[i] = glm::vec4(0.0f);
            lightVector[i] = glm::vec4(0.4f, 1.0f, 0.1f, 400.0f);
            lightOffset[i] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        bound.mn.x = 100000.f;
        bound.mn.y = 100000.f;
        bound.mn.z = 100000.f;
        bound.mn.w = 100000.f;
        bound.mx.x = -100000.f;
        bound.mx.y = -100000.f;
        bound.mx.z = -100000.f;
        bound.mx.w = -100000.f;

        arcounter = allocateBuffer<int32_t>(5);
        arcounterTemp = allocateBuffer<int32_t>(1);
        rayBlockUniform = allocateBuffer<RayBlockUniform>(1);
        lightUniform = allocateBuffer<LightUniformStruct>(6);

        glNamedBufferSubData(arcounterTemp, 0, strided<int32_t>(1), zero);

        Vc2 arr[4] = { { -1.0f, -1.0f }, { 1.0f, -1.0f },{ -1.0f, 1.0f },{ 1.0f, 1.0f } };
        glCreateBuffers(1, &posBuf);
        glNamedBufferData(posBuf, strided<Vc2>(4), arr, GL_STATIC_DRAW);

        uint32_t idc[6] = { 0, 1, 2, 3, 2, 1 };
        glCreateBuffers(1, &idcBuf);
        glNamedBufferData(idcBuf, strided<uint32_t>(6), idc, GL_STATIC_DRAW);

        glCreateVertexArrays(1, &vao);
        glVertexArrayElementBuffer(vao, idcBuf);

        glEnableVertexArrayAttrib(vao, 0);
        glVertexArrayAttribFormat(vao, 0, 2, GL_FLOAT, GL_FALSE, 0);
        glVertexArrayVertexBuffer(vao, 0, posBuf, 0, strided<Vc2>(1));

        materialUniformData.lightcount = 1;
        samplerUniformData.currentSample = currentSample;
        samplerUniformData.maxSamples = maxSamples;
        samplerUniformData.maxFilters = maxFilters;
        cameraUniformData.enable360 = 0;
        framenum = 0;
        syncUniforms();
    }


    inline void Tracer::setLightCount(size_t lightcount) {
        materialUniformData.lightcount = lightcount;
    }

    inline void Tracer::switchMode() {
        clearRays();
        clearSampler();
        cameraUniformData.enable360 = cameraUniformData.enable360 == 1 ? 0 : 1;
    }

    inline void Tracer::resize(const uint32_t & w, const uint32_t & h) {
        displayWidth = w;
        displayHeight = h;

        if (samples     != -1) glDeleteTextures(1, &samples);
        if (sampleflags != -1) glDeleteTextures(1, &sampleflags);
        if (presampled  != -1) glDeleteTextures(1, &presampled);

        samples = allocateTexture2D<GL_RGBA32F>(displayWidth, displayHeight);
        sampleflags = allocateTexture2D<GL_R32UI>(displayWidth, displayHeight);
        presampled = allocateTexture2D<GL_RGBA32F>(displayWidth, displayHeight);

        // set sampler of
        glTextureParameteri(samples, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(samples, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureParameteri(samples, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(samples, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // set sampler of
        glTextureParameteri(sampleflags, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(sampleflags, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureParameteri(sampleflags, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(sampleflags, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        // set sampler of
        glTextureParameteri(presampled, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(presampled, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureParameteri(presampled, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureParameteri(presampled, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        clearSampler();
    }

    inline void Tracer::resizeBuffers(const uint32_t & w, const uint32_t & h) {
        width = w;
        height = h;

        if (colorchains != -1) glDeleteBuffers(1, &colorchains);
        if (rays        != -1) glDeleteBuffers(1, &rays);
        if (hits        != -1) glDeleteBuffers(1, &hits);
        if (activel     != -1) glDeleteBuffers(1, &activel);
        if (activenl    != -1) glDeleteBuffers(1, &activenl);
        if (texels      != -1) glDeleteBuffers(1, &texels);
        if (freedoms    != -1) glDeleteBuffers(1, &freedoms);
        if (availables  != -1) glDeleteBuffers(1, &availables);
        if (quantized   != -1) glDeleteBuffers(1, &quantized);

        const int32_t wrsize = width * height;
        currentRayLimit = std::min(wrsize * 8, 4096 * 4096);

        colorchains = allocateBuffer<ColorChain>(currentRayLimit * 3);
        rays = allocateBuffer<Ray>(currentRayLimit);
        hits = allocateBuffer<Hit>(currentRayLimit);
        activel = allocateBuffer<int32_t>(currentRayLimit);
        activenl = allocateBuffer<int32_t>(currentRayLimit);
        texels = allocateBuffer<Texel>(wrsize);
        freedoms = allocateBuffer<int32_t>(currentRayLimit);
        availables = allocateBuffer<int32_t>(currentRayLimit);

        samplerUniformData.sceneRes = { float(width), float(height) };
        samplerUniformData.currentRayLimit = currentRayLimit;

        clearRays();
        syncUniforms();
    }

    inline void Tracer::syncUniforms() {
        for (int i = 0; i < materialUniformData.lightcount; i++) {
            lightUniformData[i].lightColor = *(Vc4 *)glm::value_ptr(lightColor[i]);
            lightUniformData[i].lightVector = *(Vc4 *)glm::value_ptr(lightVector[i]);
            lightUniformData[i].lightOffset = *(Vc4 *)glm::value_ptr(lightOffset[i]);
            lightUniformData[i].lightAmbient = *(Vc4 *)glm::value_ptr(lightAmbient[i]);
        }

        rayBlockData.cameraUniform = cameraUniformData;
        rayBlockData.samplerUniform = samplerUniformData;
        rayBlockData.materialUniform = materialUniformData;

        glNamedBufferSubData(rayBlockUniform, 0, strided<RayBlockUniform>(1), &rayBlockData);
        glNamedBufferSubData(lightUniform, 0, strided<LightUniformStruct>(6), lightUniformData);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    inline void Tracer::bindUniforms() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, lightUniform);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, rayBlockUniform);
    }

    inline void Tracer::bind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rays);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, hits);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, texels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, colorchains);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, activel);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, availables);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, activenl);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, freedoms);
        
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, arcounter);


        syncUniforms();
        bindUniforms();
    }

    inline void Tracer::clearRays() {
        bound.mn.x = 100000.f;
        bound.mn.y = 100000.f;
        bound.mn.z = 100000.f;
        bound.mn.w = 100000.f;
        bound.mx.x = -100000.f;
        bound.mx.y = -100000.f;
        bound.mx.z = -100000.f;
        bound.mx.w = -100000.f;
        for (int i = 0; i < 5;i++) {
            glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * i, sizeof(uint32_t));
        }
    }

    inline void Tracer::resetHits() {
        int32_t rsize = getRayCount();
        if (rsize <= 0) return;

        this->bind();
        dispatch(beginProgram, tiled(rsize, worksize));
    }

    inline void Tracer::sample() {
        glBindImageTexture(0, presampled, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(1, sampleflags, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);

        this->bind();
        dispatch(samplerProgram, tiled(displayWidth * displayHeight, worksize));

        currentSample = (currentSample + 1) % maxSamples;
        samplerUniformData.currentSample = currentSample;
    }

    inline void Tracer::camera(const glm::mat4 &persp, const glm::mat4 &frontSide) {
        glFlush();

        clearRays();

        materialUniformData.time = rand();
        cameraUniformData.camInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(frontSide));
        cameraUniformData.projInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(persp));
        cameraUniformData.interlace = 0;
        cameraUniformData.interlaceStage = (framenum++) % 2;

        this->bind();
        dispatch(cameraProgram, tiled(width * height, worksize));

        reloadQueuedRays(true);
    }

    inline void Tracer::camera(const glm::vec3 &eye, const glm::vec3 &view, const glm::mat4 &persp) {
#ifdef USE_CAD_SYSTEM
        glm::mat4 sidemat = glm::lookAt(eye, view, glm::vec3(0.0f, 0.0f, 1.0f));
#elif USE_180_SYSTEM
        glm::mat4 sidemat = glm::lookAt(eye, view, glm::vec3(0.0f, -1.0f, 0.0f));
#else
        glm::mat4 sidemat = glm::lookAt(eye, view, glm::vec3(0.0f, 1.0f, 0.0f));
#endif

        this->camera(persp, sidemat);
    }

    inline void Tracer::camera(const glm::vec3 &eye, const glm::vec3 &view) {
        const double PI = 3.1415926535897932384626422832795028841971;
        this->camera(eye, view, glm::perspective(float(PI) / 3.0f, float(displayWidth) / float(displayHeight), 0.001f, 1000.0f));
    }

    inline void Tracer::clearSampler() {
        samplerUniformData.samplecount = displayWidth * displayHeight;
        samplerUniformData.currentSample = 0;
        samplerUniformData.maxSamples = maxSamples;
        this->bind();

        glBindImageTexture(0, sampleflags, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
        dispatch(clearProgram, tiled(displayWidth * displayHeight, worksize));
    }

    inline void Tracer::reloadQueuedRays(bool doSort, bool sortMortons) {
        glGetNamedBufferSubData(arcounter, 0 * sizeof(uint32_t), sizeof(uint32_t), &raycountCache);
        samplerUniformData.rayCount = raycountCache;
        syncUniforms();

        uint32_t availableCount = 0;
        glGetNamedBufferSubData(arcounter, 2 * sizeof(int32_t), sizeof(int32_t), &availableCount);

        // set available pointer
        int32_t availablePtr = int32_t(availableCount) - 1;
        //glNamedBufferSubData(arcounter, 3 * sizeof(int32_t), sizeof(int32_t), &availablePtr);

        // copy pointer+1
        glCopyNamedBufferSubData(arcounter, arcounter, 2 * sizeof(int32_t), 3 * sizeof(int32_t), sizeof(int32_t));

        // set to zeros
        glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * 2, sizeof(uint32_t));
        glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * 0, sizeof(uint32_t));

        // copy active collection
        if (raycountCache > 0) {
            glCopyNamedBufferSubData(activenl, activel, 0, 0, strided<uint32_t>(raycountCache));
        }

        // copy collection of available ray memory 
        if (availableCount > 0) {
            glCopyNamedBufferSubData(freedoms, availables, 0, 0, strided<uint32_t>(availableCount));
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // sort actives by index
        if (raycountCache > 0 && doSort) {
            //sorter->sort(activel, activel, raycountCache);
        }
    }

    inline void Tracer::reclaim() {
        int32_t rsize = getRayCount();
        if (rsize <= 0) return;

        this->bind();

        dispatch(reclaimProgram, tiled(rsize, worksize));
        reloadQueuedRays(true);
    }

    inline void Tracer::render() {
        this->bind();
        glEnable(GL_TEXTURE_2D);
        glBindTextureUnit(5, presampled);
        glScissor(0, 0, displayWidth, displayHeight);
        glViewport(0, 0, displayWidth, displayHeight);
        glUseProgram(renderProgram);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glBindVertexArray(0);
    }

    inline int Tracer::intersection(Intersector * obj, const int clearDepth) {
        if (!obj || obj->triangleCount <= 0) return 0;

        int32_t rsize = getRayCount();
        if (rsize <= 0) return 0;

        bound.mn = glm::min(obj->bound.mn, bound.mn);
        bound.mx = glm::max(obj->bound.mx, bound.mx);

        obj->configureIntersection(clearDepth);
        obj->bind();
        obj->bindBVH();
        this->bind();
        dispatch(intersectionProgram, tiled(rsize, worksize));

        return 1;
    }

    inline void Tracer::shade(Material * mat) {
        int32_t rsize = getRayCount();
        if (rsize <= 0) return;

        samplerUniformData.rayCount = rsize;
        materialUniformData.time = rand();

        this->bind();
        mat->bindWithContext(matProgram);

        glBindTextureUnit(5, skybox);

        dispatch(matProgram, tiled(rsize, worksize));
    }

    inline int32_t Tracer::getRayCount() {
        return raycountCache >= 32 ? raycountCache : 0;
    }
}
