#include "Pipeline.hpp"

// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md

namespace NSM {

    inline Pipeline::~Pipeline(){
        
        glDeleteProgram(renderProgram);
        glDeleteProgram(matProgram);
        glDeleteProgram(reclaimProgram);
        glDeleteProgram(cameraProgram);
        glDeleteProgram(clearProgram);
        glDeleteProgram(samplerProgram);
        glDeleteProgram(traverseProgram);
        glDeleteProgram(resolverProgram);
        glDeleteProgram(surfProgram);
        glDeleteProgram(filterProgram);

        glDeleteBuffers(1, &colorchains);
        glDeleteBuffers(1, &quantized);
        glDeleteBuffers(1, &rays);
        glDeleteBuffers(1, &hits);
        glDeleteBuffers(1, &texels);
        glDeleteBuffers(1, &activenl);
        glDeleteBuffers(1, &activel);
        glDeleteBuffers(1, &freedoms);
        glDeleteBuffers(1, &availables);
        glDeleteBuffers(1, &arcounter);
        glDeleteBuffers(1, &arcounterTemp);

        glDeleteBuffers(1, &lightUniform);
        glDeleteBuffers(1, &rayBlockUniform);

        glDeleteTextures(1, &presampled);
        glDeleteTextures(1, &sampleflags);

        glDeleteVertexArrays(1, &vao);
        
    }

    inline void Pipeline::initShaders() {
        
        initShaderComputeSPIRV("./shaders-spv/raytracing/surface.comp.spv", surfProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/testmat.comp.spv", matProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/reclaim.comp.spv", reclaimProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/camera.comp.spv", cameraProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/clear.comp.spv", clearProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/sampler.comp.spv", samplerProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/filter.comp.spv", filterProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/directTraverse.comp.spv", traverseDirectProgram);
        initShaderComputeSPIRV("./shaders-spv/raytracing/deinterlace.comp.spv", deinterlaceProgram);
        

        {
            renderProgram = glCreateProgram();
            glAttachShader(renderProgram, loadShaderSPIRV("./shaders-spv/raytracing/render.vert.spv", GL_VERTEX_SHADER));
            glAttachShader(renderProgram, loadShaderSPIRV("./shaders-spv/raytracing/render.frag.spv", GL_FRAGMENT_SHADER));
            glLinkProgram(renderProgram);
            validateProgram(renderProgram);
        }
    }




    inline void Pipeline::initVAO() {
        Vc2 arr[4] = { { -1.0f, -1.0f },{ 1.0f, -1.0f },{ -1.0f, 1.0f },{ 1.0f, 1.0f } };

		GLuint posBuf, idcBuf;

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
    }


    inline void Pipeline::init() {
        initShaders();
        initVAO();

        lightUniformData = new LightUniformStruct[6];
        for (int i = 0; i < 6;i++) {
            lightColor[i] = glm::vec4((glm::vec3(255.f, 250.f, 244.f) / 255.f) * 50.f, 40.0f);
            lightAmbient[i] = glm::vec4(0.0f);
            lightVector[i] = glm::vec4(0.3f, 1.0f, 0.1f, 400.0f);
            lightOffset[i] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        bbox bound;
        bound.mn.x = 100000.f;
        bound.mn.y = 100000.f;
        bound.mn.z = 100000.f;
        bound.mn.w = 100000.f;
        bound.mx.x = -100000.f;
        bound.mx.y = -100000.f;
        bound.mx.z = -100000.f;
        bound.mx.w = -100000.f;
        framenum = 0;

        resultCounters = allocateBuffer<uint32_t>(2);
        arcounter = allocateBuffer<int32_t>(8);
        arcounterTemp = allocateBuffer<int32_t>(1);
        rayBlockUniform = allocateBuffer<RayBlockUniform>(1);
        lightUniform = allocateBuffer<LightUniformStruct>(6);
        glNamedBufferSubData(arcounterTemp, 0, strided<int32_t>(1), zero);

        materialUniformData.lightcount = 1;
        cameraUniformData.enable360 = 0;
        syncUniforms();
    }


    inline void Pipeline::setLightCount(size_t lightcount) {
        materialUniformData.lightcount = lightcount;
    }

    inline void Pipeline::switchMode() {
        clearRays();
        clearSampler();
        cameraUniformData.enable360 = cameraUniformData.enable360 == 1 ? 0 : 1;
    }

    inline void Pipeline::resize(const uint32_t & w, const uint32_t & h) {
        displayWidth = w;
        displayHeight = h;

        if (sampleflags != -1) glDeleteTextures(1, &sampleflags);
        if (presampled  != -1) glDeleteTextures(1, &presampled);
        if (filtered != -1) glDeleteTextures(1, &filtered);
        //if (prevsampled != -1) glDeleteTextures(1, &prevsampled);
        //if (positionimg != -1) glDeleteTextures(1, &positionimg);

        //reprojected = allocateTexture2D<GL_RGBA32F>(displayWidth, displayHeight);
        //positionimg = allocateTexture2D<GL_RGBA32F>(displayWidth, displayHeight);

        sampleflags = allocateTexture2D<GL_R32UI>(displayWidth, displayHeight);
        presampled = allocateTexture2D<GL_RGBA32F>(displayWidth, displayHeight);
		filtered = allocateTexture2D<GL_RGBA32F>(displayWidth, displayHeight);
        //prevsampled = allocateTexture2D<GL_RGBA32F>(displayWidth, displayHeight);
        
        
        // set sampler of
        glTextureParameteri(presampled, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(presampled, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // previous frame for temporal AA
        //glTextureParameteri(prevsampled, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        //glTextureParameteri(prevsampled, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        // set sampler of
        glTextureParameteri(filtered, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(filtered, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        samplerUniformData.samplecount = displayWidth * displayHeight;
        clearSampler();
    }

    inline void Pipeline::resizeBuffers(const uint32_t & w, const uint32_t & h) {
        width = w, height = h;
        bool enableInterlacing = false;

        if (colorchains   != -1) glDeleteBuffers(1, &colorchains);
        if (rays          != -1) glDeleteBuffers(1, &rays);
        if (hits          != -1) glDeleteBuffers(1, &hits);
        if (activel       != -1) glDeleteBuffers(1, &activel);
        if (activenl      != -1) glDeleteBuffers(1, &activenl);
        if (texels        != -1) glDeleteBuffers(1, &texels);
        if (freedoms      != -1) glDeleteBuffers(1, &freedoms);
        if (availables    != -1) glDeleteBuffers(1, &availables);
        if (quantized     != -1) glDeleteBuffers(1, &quantized);
        if (deferredStack != -1) glDeleteBuffers(1, &deferredStack);

        //const int32_t cmultiplier = 6;
		const int32_t cmultiplier = 4;
        const int32_t wrsize = width * height;
        currentRayLimit = std::min(wrsize * cmultiplier / (enableInterlacing ? 2 : 1), 4096 * 4096);

        colorchains = allocateBuffer<ColorChain>(wrsize * 8);
		texels = allocateBuffer<Texel>(wrsize);

        rays = allocateBuffer<Ray>(currentRayLimit);
        hits = allocateBuffer<Hit>(currentRayLimit / 2);
        activel = allocateBuffer<int32_t>(currentRayLimit);
        activenl = allocateBuffer<int32_t>(currentRayLimit);
        freedoms = allocateBuffer<int32_t>(currentRayLimit);
        availables = allocateBuffer<int32_t>(currentRayLimit);
        deferredStack = allocateBuffer<int32_t>(currentRayLimit * 8);

        samplerUniformData.sceneRes = { float(width), float(height) };
        samplerUniformData.currentRayLimit = currentRayLimit;
        cameraUniformData.interlace = enableInterlacing ? 1 : 0;

        clearRays();
        syncUniforms();
    }

    inline void Pipeline::syncUniforms() {
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

    inline void Pipeline::bindUniforms() {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, lightUniform);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, rayBlockUniform);
    }

    inline void Pipeline::bind() {
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

    inline void Pipeline::clearRays() {
        for (int i = 0; i < 8;i++) {
            glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * i, sizeof(uint32_t));
        }
    }

    inline void Pipeline::sample() {
        
        // collect samples
        glBindImageTexture(0, presampled, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(1, sampleflags, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
        this->bind();
        dispatch(samplerProgram, tiled(displayWidth * displayHeight, worksize));

        // filter by deinterlacing, etc.
        glBindImageTexture(0, presampled, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        glBindImageTexture(1, filtered, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
        //glBindImageTexture(2, prevsampled, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

        // deinterlace if need
        dispatch(deinterlaceProgram, tiled(displayWidth * displayHeight, worksize));

        // use temporal AA
        dispatch(filterProgram, tiled(displayWidth * displayHeight, worksize));

        // save previous frame
        //glCopyImageSubData(filtered, GL_TEXTURE_2D, 0, 0, 0, 0, prevsampled, GL_TEXTURE_2D, 0, 0, 0, 0, displayWidth, displayHeight, 1);
        
    }

    inline void Pipeline::camera(const glm::mat4 &persp, const glm::mat4 &frontSide) {
        clearRays();

        materialUniformData.time = rand();
        cameraUniformData.camInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(frontSide));
        cameraUniformData.projInv = *(Vc4x4 *)glm::value_ptr(glm::inverse(persp));
        cameraUniformData.interlaceStage = (framenum++) % 2;

        this->bind();
        dispatch(cameraProgram, tiled(width * height, worksize));

        reloadQueuedRays(true);
    }

    inline void Pipeline::camera(const glm::vec3 &eye, const glm::vec3 &view, const glm::mat4 &persp) {
#ifdef USE_CAD_SYSTEM
        glm::mat4 sidemat = glm::lookAt(eye, view, glm::vec3(0.0f, 0.0f, 1.0f));
#elif USE_180_SYSTEM
        glm::mat4 sidemat = glm::lookAt(eye, view, glm::vec3(0.0f, -1.0f, 0.0f));
#else
        glm::mat4 sidemat = glm::lookAt(eye, view, glm::vec3(0.0f, 1.0f, 0.0f));
#endif

        this->camera(persp, sidemat);
    }

    inline void Pipeline::camera(const glm::vec3 &eye, const glm::vec3 &view) {
        this->camera(eye, view, glm::perspective(glm::pi<float>() / 3.0f, float(displayWidth) / float(displayHeight), 0.001f, 1000.0f));
    }

    inline void Pipeline::clearSampler() {
        this->bind();

        glBindImageTexture(0, sampleflags, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
        dispatch(clearProgram, tiled(displayWidth * displayHeight, worksize));
    }

    inline void Pipeline::reloadQueuedRays(bool doSort, bool sortMortons) {
		int32_t counters[8];
		counters[3] = counters[3] >= 0 ? counters[3] : 0;
		glGetNamedBufferSubData(arcounter, 0 * sizeof(uint32_t), 8 * sizeof(uint32_t), counters);

		raycountCache = counters[0];
        samplerUniformData.rayCount = raycountCache;
        //syncUniforms(); // no need extra loud

        uint32_t availableCount = counters[2];
        glCopyNamedBufferSubData(arcounter, arcounter, 2 * sizeof(int32_t), 3 * sizeof(int32_t), sizeof(int32_t));
		//glNamedBufferSubData(arcounterTemp, 3 * sizeof(int32_t), sizeof(int32_t), &availableCountLeast);

        // set to zeros
        glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * 7, sizeof(uint32_t));
        //glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * 2, sizeof(uint32_t));
		glNamedBufferSubData(arcounterTemp, 2 * sizeof(int32_t), sizeof(int32_t), &counters[3]); // using with least
        glCopyNamedBufferSubData(arcounterTemp, arcounter, 0, sizeof(uint32_t) * 0, sizeof(uint32_t));

        // copy active collection
        if (raycountCache > 0) {
            //glCopyNamedBufferSubData(activenl, activel, 0, 0, strided<uint32_t>(raycountCache));
			SWAP(activenl, activel);
        }

        // copy collection of available ray memory 
        if (availableCount > 0) {
            //glCopyNamedBufferSubData(freedoms, availables, 0, 0, strided<uint32_t>(availableCount));
			if (counters[3] > 0) glCopyNamedBufferSubData(availables, freedoms, 0, 0, strided<uint32_t>(counters[3])); // copy free memory data
			SWAP(freedoms, availables);
        }

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
    }

    inline void Pipeline::reclaim() {
        //int32_t rsize = getRayCount();
        //if (rsize <= 0) return;

        //this->bind();

        //dispatch(reclaimProgram, tiled(rsize, worksize));
        //reloadQueuedRays(true);
    }

    inline void Pipeline::render() {
        this->bind();
        glEnable(GL_TEXTURE_2D);
        glBindTextureUnit(5, filtered);
        //glBindTextureUnit(6, prevsampled);
        glScissor(0, 0, displayWidth, displayHeight);
        glViewport(0, 0, displayWidth, displayHeight);
        glUseProgram(renderProgram);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glBindVertexArray(0);
    }

    inline int Pipeline::intersection(TriangleHierarchy * obj, const int clearDepth) {
        if (!obj || obj->triangleCount <= 0) return 0;

        int32_t rsize = getRayCount();
        if (rsize <= 0) return 0;

        this->bind();
        obj->bindBVH();
        obj->bindLeafs();
        obj->bind();

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 19, deferredStack);
        dispatch(traverseDirectProgram, tiled(rsize, worksize));
        
        return 1;
    }

    inline void Pipeline::applyMaterials(MaterialSet * mat) {
        mat->bindWithContext(surfProgram);
        glCopyNamedBufferSubData(arcounter, rayBlockUniform, 7 * sizeof(int32_t), offsetof(RayBlockUniform, samplerUniform) + offsetof(SamplerUniformStruct, hitCount), sizeof(int32_t));
        GLuint tcount = 0; glGetNamedBufferSubData(arcounter, 7 * sizeof(int32_t), sizeof(int32_t), &tcount);
        if (tcount <= 0) return;
        dispatch(surfProgram, tiled(tcount, worksize));
    }

    inline void Pipeline::shade() {
        int32_t rsize = getRayCount();
        if (rsize <= 0) return;
        materialUniformData.time = rand(); this->bind();
        glBindTextureUnit(5, skybox);
        dispatch(matProgram, tiled(rsize, worksize));
		reloadQueuedRays(true); // you can at now
    }


    inline Pipeline::HdrImage Pipeline::snapRawHdr() {
        HdrImage img;
        img.width = displayWidth;
        img.height = displayHeight;
        img.image = new GLfloat[displayWidth * displayHeight * 4];
        glGetTextureSubImage(presampled, 0, 0, 0, 0, displayWidth, displayHeight, 1, GL_RGBA, GL_FLOAT, displayWidth * displayHeight * 4 * sizeof(GLfloat), img.image);
        return img;
    }

    inline Pipeline::HdrImage Pipeline::snapHdr() {
        HdrImage img;
        img.width = displayWidth;
        img.height = displayHeight;
        img.image = new GLfloat[displayWidth * displayHeight * 4];
        glGetTextureSubImage(filtered, 0, 0, 0, 0, displayWidth, displayHeight, 1, GL_RGBA, GL_FLOAT, displayWidth * displayHeight * 4 * sizeof(GLfloat), img.image);
        return img;
    }



    inline int32_t Pipeline::getRayCount() {
        return raycountCache >= 32 ? raycountCache : 0;
    }
}
