#include "tracer.hpp"

namespace Paper {

    inline void Tracer::initShaderCompute(std::string path, GLuint & prog) {
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

                std::cout << str << std::endl;

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

    inline void Tracer::initShaders() {
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

        glCreateBuffers(1, &arcounter);
        glNamedBufferStorage(arcounter, strided<int32_t>(4), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &arcounterTemp);
        glNamedBufferStorage(arcounterTemp, strided<int32_t>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferSubData(arcounterTemp, 0, strided<int32_t>(1), zero);

        glCreateBuffers(1, &rayBlockUniform);
        glNamedBufferStorage(rayBlockUniform, strided<RayBlockUniform>(1), nullptr, GL_DYNAMIC_STORAGE_BIT);

        glCreateBuffers(1, &lightUniform);
        glNamedBufferStorage(lightUniform, strided<LightUniformStruct>(6), nullptr, GL_DYNAMIC_STORAGE_BIT);

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

        materialUniformData.f_reflections = 1;
        materialUniformData.f_shadows = 1;
        materialUniformData.lightcount = 1;
        samplerUniformData.currentSample = currentSample;
        samplerUniformData.maxSamples = maxSamples;
        samplerUniformData.maxFilters = maxFilters;
        cameraUniformData.enable360 = 0;
        framenum = 0;
        syncUniforms();
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

        glCreateTextures(GL_TEXTURE_2D, 1, &samples);
        glCreateTextures(GL_TEXTURE_2D, 1, &sampleflags);
        glCreateTextures(GL_TEXTURE_2D, 1, &presampled);

        glTextureStorage2D(samples, 1, GL_RGBA32F, displayWidth, displayHeight);
        glTextureStorage2D(sampleflags, 1, GL_R32UI, displayWidth, displayHeight);
        glTextureStorage2D(presampled, 1, GL_RGBA32F, displayWidth, displayHeight);

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

        if (rays       != -1) glDeleteBuffers(1, &rays);
        if (hits       != -1) glDeleteBuffers(1, &hits);
        if (activel    != -1) glDeleteBuffers(1, &activel);
        if (activenl   != -1) glDeleteBuffers(1, &activenl);
        if (texels     != -1) glDeleteBuffers(1, &texels);
        if (freedoms   != -1) glDeleteBuffers(1, &freedoms);
        if (availables != -1) glDeleteBuffers(1, &availables);

        const int32_t wrsize = width * height;
        currentRayLimit = std::min(wrsize * 4, 4096 * 4096);

        glCreateBuffers(1, &rays);
        glCreateBuffers(1, &hits);
        glCreateBuffers(1, &activel);
        glCreateBuffers(1, &activenl);
        glCreateBuffers(1, &texels);
        glCreateBuffers(1, &freedoms);
        glCreateBuffers(1, &availables);

        glNamedBufferStorage(rays, strided<Ray>(currentRayLimit), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferStorage(hits, strided<Hit>(currentRayLimit), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferStorage(activel, strided<int32_t>(currentRayLimit), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferStorage(activenl, strided<int32_t>(currentRayLimit), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferStorage(texels, strided<Texel>(wrsize), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferStorage(freedoms, strided<int32_t>(currentRayLimit), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glNamedBufferStorage(availables, strided<int32_t>(currentRayLimit), nullptr, GL_DYNAMIC_STORAGE_BIT);

        samplerUniformData.sceneRes = { float(width), float(height) };
        samplerUniformData.currentRayLimit = currentRayLimit;

        clearRays();
        syncUniforms();
    }

    inline void Tracer::enableReflections(const int32_t flag) {
        materialUniformData.f_reflections = flag;
    }

    inline void Tracer::enableShadows(const int32_t flag) {
        materialUniformData.f_shadows = flag;
    }

    inline void Tracer::includeCubemap(GLuint cube) { 
        cubeTex = cube; 
    }

    inline void Tracer::syncUniforms() {
        for (int i = 0; i < materialUniformData.lightcount; i++) {
            lightUniformData[i].lightColor = *(Vc4 *)glm::value_ptr(lightColor[i]);
            lightUniformData[i].lightVector = *(Vc4 *)glm::value_ptr(lightVector[i]);
            lightUniformData[i].lightOffset = *(Vc4 *)glm::value_ptr(lightOffset[i]);
        }

        rayBlockData.cameraUniform = cameraUniformData;
        rayBlockData.samplerUniform = samplerUniformData;
        rayBlockData.materialUniform = materialUniformData;
        rayBlockData.randomUniform = randomUniformData;

        glNamedBufferSubData(rayBlockUniform, 0, strided<RayBlockUniform>(1), &rayBlockData);
        glNamedBufferSubData(lightUniform, 0, strided<LightUniformStruct>(6), lightUniformData);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    inline void Tracer::bindUniforms() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, lightUniform);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, rayBlockUniform);
    }

    inline void Tracer::bind() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, arcounter);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rays);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, hits);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, texels);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, activel);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, activenl);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, freedoms);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, availables);

        syncUniforms();
        bindUniforms();
    }

    inline void Tracer::clearRays() {
        for (int i = 0; i < 4;i++) {
            glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * i, sizeof(uint32_t));
        }
    }

    inline void Tracer::resetHits() {
        int32_t rsize = getRayCount();
        if (rsize <= 0) return;

        this->bind();

        glUseProgram(beginProgram);
        glDispatchCompute(tiled(rsize, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    inline void Tracer::sample() {
        glBindImageTexture(0, sampleflags, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
        glBindImageTexture(1, samples, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(2, presampled, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

        this->bind();

        glUseProgram(samplerProgram);
        glDispatchCompute(tiled(displayWidth * displayHeight, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        currentSample = (currentSample + 1) % maxSamples;
        samplerUniformData.currentSample = currentSample;
    }

    inline void Tracer::camera(const glm::mat4 &persp, const glm::mat4 &frontSide) {
        glFlush();

        clearRays();

        randomUniformData.time = frandom();
        cameraUniformData.camInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(frontSide));
        cameraUniformData.projInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(persp));
        cameraUniformData.interlace = 1;
        cameraUniformData.interlaceStage = (framenum++) % 2;

        this->bind();
        glUseProgram(cameraProgram);
        glDispatchCompute(tiled(width * height, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

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
        this->camera(eye, view, glm::perspective(((float)M_PI / 3.0f), (float)displayWidth / (float)displayHeight, 0.001f, 1000.0f));
    }

    inline void Tracer::clearSampler() {
        samplerUniformData.samplecount = displayWidth * displayHeight;
        samplerUniformData.currentSample = 0;
        samplerUniformData.maxSamples = maxSamples;
        this->bind();

        glBindImageTexture(0, sampleflags, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
        glUseProgram(clearProgram);
        glDispatchCompute(tiled(displayWidth * displayHeight, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    inline void Tracer::reloadQueuedRays(bool doSort) {
        glGetNamedBufferSubData(arcounter, 0 * sizeof(uint32_t), sizeof(uint32_t), &raycountCache);
        samplerUniformData.rayCount = raycountCache;
        syncUniforms();

        uint32_t availableCount = 0;
        glGetNamedBufferSubData(arcounter, 2 * sizeof(int32_t), 1 * sizeof(int32_t), &availableCount);
        glNamedBufferSubData(arcounter, 3 * sizeof(int32_t), 1 * sizeof(int32_t), &availableCount);
        glNamedBufferSubData(arcounter, 2 * sizeof(int32_t), 1 * sizeof(int32_t), zero);
        glNamedBufferSubData(arcounter, 0 * sizeof(int32_t), 1 * sizeof(int32_t), zero);

        if (raycountCache > 0) {
            glCopyNamedBufferSubData(activenl, activel, 0, 0, strided<uint32_t>(raycountCache));
        }

        if (availableCount > 0) {
            glCopyNamedBufferSubData(freedoms, availables, 0, 0, strided<uint32_t>(availableCount));
        }
        glMemoryBarrier(GL_ALL_BARRIER_BITS);

        // sort actives by index
        if (raycountCache > 0 && doSort) {
            sorter->sort(activel, activel, raycountCache);
        }

        // TODO sort by quantization
    }

    inline void Tracer::reclaim() {
        int32_t rsize = getRayCount();
        if (rsize <= 0) return;

        this->bind();

        glUseProgram(reclaimProgram);
        glDispatchCompute(tiled(rsize, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        reloadQueuedRays(true);
    }

    inline void Tracer::render() {
        this->bind();
        glEnable(GL_TEXTURE_2D);
        glBindTextureUnit(0, presampled);
        glScissor(0, 0, displayWidth, displayHeight);
        glViewport(0, 0, displayWidth, displayHeight);
        glUseProgram(renderProgram);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glBindVertexArray(0);
        glFlush();
    }

    inline int Tracer::intersection(Intersector * obj, const int clearDepth) {
        if (!obj || obj->triangleCount <= 0) return 0;

        int32_t rsize = getRayCount();
        if (rsize <= 0) return 0;

        obj->configureIntersection(clearDepth);
        obj->bind();
        obj->bindBVH();
        this->bind();

        const size_t worksize = 64;
        glUseProgram(intersectionProgram);
        glDispatchCompute(tiled(rsize, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        return 1;
    }

    inline void Tracer::shade(Material * mat) {
        int32_t rsize = getRayCount();
        if (rsize <= 0) return;

        samplerUniformData.rayCount = rsize;
        randomUniformData.time = frandom();

        this->bind();
        if (cubeTex) glBindTextureUnit(0, cubeTex);

        glUseProgram(matProgram);
        mat->bindWithContext(matProgram);
        glDispatchCompute(tiled(rsize, worksize), 1, 1);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        reloadQueuedRays();
    }

    inline int32_t Tracer::getRayCount() {
        return raycountCache >= 32 ? raycountCache : 0;
    }
}
