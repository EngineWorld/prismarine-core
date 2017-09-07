#pragma once

#include "Utils.hpp"
#include "Structs.hpp"
#include "SceneObject.hpp"
#include "MaterialSet.hpp"

namespace ppr {

    class Dispatcher : public BaseClass {
    protected:
        RadixSort * sorter = nullptr;

        GLuint skybox = -1;

        GLuint renderProgram = -1;
        GLuint surfProgram = -1;
        GLuint matProgram = -1;
        GLuint reclaimProgram = -1;
        GLuint cameraProgram = -1;
        GLuint clearProgram = -1;
        GLuint samplerProgram = -1;
        GLuint traverseProgram = -1;
        GLuint resolverProgram = -1;
        GLuint traverseDirectProgram = -1;
        GLuint filterProgram = -1;
        GLuint deinterlaceProgram = -1;

        GLuint colorchains = -1;
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
        GLuint hitChains = -1;
        GLuint deferredStack = -1;

        GLuint lightUniform = -1;
        GLuint rayBlockUniform = -1;
        RayBlockUniform rayBlockData;

        size_t framenum = 0;
        int32_t currentRayLimit = 0;
        int32_t worksize = 128;

        // position texture
        GLuint positionimg = -1;
        GLuint prevsampled = -1;

        // frame buffers
        GLuint presampled = -1;
        GLuint reprojected = -1;
        GLuint sampleflags = -1;
        GLuint filtered = -1;

        GLuint vao = -1;
        //GLuint posBuf = -1;
        //GLuint idcBuf = -1;

        void initShaders();
        void initVAO();
        void init();

        MaterialUniformStruct materialUniformData;
        SamplerUniformStruct samplerUniformData;
        CameraUniformStruct cameraUniformData;

        GLuint resultCounters = -1;
        GLuint resultFounds = -1;
        GLuint givenRays = -1;

    public:

        Dispatcher() { init(); }
        ~Dispatcher();

        uint32_t width = 256;
        uint32_t height = 256;
        uint32_t displayWidth = 256;
        uint32_t displayHeight = 256;


        void setSkybox(GLuint skb) {
            skybox = skb;
        }

        void switchMode();
        void resize(const uint32_t & w, const uint32_t & h);
        void resizeBuffers(const uint32_t & w, const uint32_t & h);
        void syncUniforms();
        void reloadQueuedRays(bool doSort = false, bool sortMortons = false);

        LightUniformStruct * lightUniformData;
        int32_t raycountCache = 0;
        int32_t qraycountCache = 0;

        glm::vec4 lightColor[6];
        glm::vec4 lightAmbient[6];
        glm::vec4 lightVector[6];
        glm::vec4 lightOffset[6];

        struct HdrImage {
            GLfloat * image = nullptr;
            int width = 1;
            int height = 1;
        };

        HdrImage snapHdr();
        HdrImage snapRawHdr();

        void setLightCount(size_t lightcount);
        void bindUniforms();
        void bind();
        void clearRays();
        void sample();
        void camera(const glm::mat4 &persp, const glm::mat4 &frontSide);
        void camera(const glm::vec3 &eye, const glm::vec3 &view, const glm::mat4 &persp);
        void camera(const glm::vec3 &eye, const glm::vec3 &view);
        void clearSampler();
        void reclaim();
        void render();
        int intersection(SceneObject * obj, const int clearDepth = 0);
        void shade();
        void applyMaterials(MaterialSet * mat);
        int32_t getRayCount();
    };
}
