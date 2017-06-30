#pragma once

#include "includes.hpp"
#include "utils.hpp"
#include "structs.hpp"
#include "intersector.hpp"
#include "material.hpp"
#include <cmath>

namespace Paper {

    class Tracer : public PTObject {
    private:
        RadixSort * sorter;

        GLuint renderProgram = -1;
        GLuint matProgram = -1;
        GLuint beginProgram = -1;
        GLuint reclaimProgram = -1;
        GLuint cameraProgram = -1;
        GLuint clearProgram = -1;
        GLuint samplerProgram = -1;
        GLuint intersectionProgram = -1;

        GLuint quantized = -1;
        GLuint rays = -1;
        GLuint hits = -1;
        GLuint texels = -1;
        GLuint activenl = -1;
        GLuint activel = -1;
        GLuint freedoms = -1;
        GLuint availables = -1;
        GLuint arcounter = -1;
        GLuint arcounterTemp = -1;

        GLuint lightUniform = -1;
        GLuint rayBlockUniform = -1;
        RayBlockUniform rayBlockData;

        size_t framenum = 0;
        int32_t currentSample = 0;
        int32_t maxSamples = 4;
        int32_t maxFilters = 1;
        int32_t currentRayLimit = 0;
        int32_t worksize = 128;

        GLuint presampled = -1;
        GLuint samples = -1;
        GLuint sampleflags = -1;
        GLuint cubeTex = -1;
        GLuint vao = -1;

        GLuint posBuf = -1;
        GLuint idcBuf = -1;
        GLuint posattr = -1;

        const int32_t zero[1] = { 0 };
        void initShaderCompute(std::string str, GLuint& prog);
        void initShaders();
        void init();

        RandomUniformStruct randomUniformData;
        MaterialUniformStruct materialUniformData;
        SamplerUniformStruct samplerUniformData;
        CameraUniformStruct cameraUniformData;

        bbox bound;

    public:
        Tracer() { init(); }

        uint32_t width = 256;
        uint32_t height = 256;
        uint32_t displayWidth = 256;
        uint32_t displayHeight = 256;

        void switchMode();
        void resize(const uint32_t & w, const uint32_t & h);
        void resizeBuffers(const uint32_t & w, const uint32_t & h);
        void syncUniforms();
        void reloadQueuedRays(bool doSort = false, bool sortMortons = false);

        LightUniformStruct * lightUniformData;
        int32_t raycountCache = 0;
        int32_t qraycountCache = 0;

        glm::vec4 lightColor[6] = { glm::vec4((glm::vec3(255.f, 241.f, 224.f) / 255.f) * 150.f, 40.0f) };
        glm::vec4 lightVector[6] = { glm::vec4(0.4f, 1.0f, 0.1f, 400.0f) };
        glm::vec4 lightOffset[6] = { glm::vec4(0.0f, 0.0f, 0.0f, 0.0f) };

        void enableReflections(const int32_t flag);
        void enableShadows(const int32_t flag);
        void includeCubemap(GLuint cube);
        void bindUniforms();
        void bind();
        void clearRays();
        void resetHits();
        void sample();
        void camera(const glm::mat4 &persp, const glm::mat4 &frontSide);
        void camera(const glm::vec3 &eye, const glm::vec3 &view, const glm::mat4 &persp);
        void camera(const glm::vec3 &eye, const glm::vec3 &view);
        void clearSampler();
        void reclaim();
        void render();
        int intersection(Intersector * obj, const int clearDepth = 0);
        void shade(Material * mat);
        int32_t getRayCount();
    };
}

#include "./tracer.inl"