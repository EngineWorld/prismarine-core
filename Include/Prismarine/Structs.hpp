#pragma once

#include "Utils.hpp"

namespace ppr {
    
    typedef float Vc1;
    typedef int32_t iVc1;

    struct Vc2 {
        float x;
        float y;
    };

    struct Vc3 {
        float x;
        float y;
        float z;
    };

    struct Vc4 {
        float x;
        float y;
        float z;
        float w;
    };

    struct Vc4x4 {
        Vc4 m0;
        Vc4 m1;
        Vc4 m2;
        Vc4 m3;
    };


    struct iVc2 {
        int32_t x;
        int32_t y;
    };

    struct iVc3 {
        int32_t x;
        int32_t y;
        int32_t z;
    };

    struct iVc4 {
        int32_t x;
        int32_t y;
        int32_t z;
        int32_t w;
    };



    const Vc4x4 mat4r = {
        { 1.0f, 0.0f, 0.0f, 0.0f },
        { 0.0f, 1.0f, 0.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f, 0.0f },
        { 0.0f, 0.0f, 0.0f, 1.0f },
    };

    const Vc4 vec4r = { 0.0f, 0.0f, 0.0f, 0.0f };
    const iVc4 ivec4r = { 0, 0, 0, 0 };

    struct Minmax  {
        Vc4 mn;
        Vc4 mx;
    };

    struct Minmaxi {
        iVc4 mn;
        iVc4 mx;
    };

    //typedef Minmax bbox;

    struct bbox {
        glm::vec4 mn;
        glm::vec4 mx;
    };


    struct Ray {
        glm::vec4 origin;
        glm::vec4 direct;
        glm::vec4 color;
        glm::vec4 final;
        int bitfield; // up to 32-bits
        int idx; // ray index itself
        int texel; // texel index
        int hit; // index of hit chain
    };

    struct Hit {
        glm::vec4 uvt; // UV, distance, triangle
        glm::vec4 albedo;
        glm::vec4 metallicRoughness; // Y - roughtness, Z - metallic, also available other params
        glm::vec4 normalHeight; // normal with height mapping, will already interpolated with geometry
        glm::vec4 emission;
        glm::vec4 texcoord;
        glm::vec4 tangent;
        int bitfield;
        int ray; // ray index
        int materialID;
        int next;
    };


    struct Texel {
        Vc4 coord;
        iVc4 EXT;
    };

    struct HlbvhNode {
        bbox box;
        iVc4 pdata;
    };

    struct VboDataStride {
        Vc4 vertex;
        Vc4 normal;
        Vc4 texcoord;
        Vc4 color;
        Vc4 modifiers;
    };

    struct ColorChain {
        Vc4 color = {0.0f, 0.0f, 0.0f, 0.0f};
        iVc4 cdata = {0, 0, 0, 0};
    };



    struct MaterialUniformStruct {
        iVc1 materialID;
        iVc1 _reserved;
        iVc1 time;
        iVc1 lightcount;
        Vc4 backgroundColor = vec4r; // for skybox configure
        iVc4 iModifiers0 = ivec4r;
        iVc4 iModifiers1 = ivec4r;
        Vc4 fModifiers0 = vec4r;
        Vc4 fModifiers1 = vec4r;
        Vc4x4 transformModifier = mat4r;
    };

    struct SamplerUniformStruct {
        Vc2 sceneRes;
        iVc1 samplecount;
        iVc1 rayCount;
        iVc1 iteration;
        iVc1 phase;
        iVc1 hitCount;
        iVc1 reserved0;
        iVc1 reserved1;
        iVc1 currentRayLimit;

        iVc2 padding;
    };

    struct LightUniformStruct {
        Vc4 lightVector;
        Vc4 lightColor;
        Vc4 lightOffset;
        Vc4 lightAmbient;
    };

    struct GeometryUniformStruct {
        Vc4x4 transform = mat4r;
        Vc4x4 transformInv = mat4r;

        iVc1 materialID = 0;
        iVc1 triangleCount = 1;
        iVc1 triangleOffset = 0;
        iVc1 clearDepth = 0;
    };

    struct CameraUniformStruct {
        Vc4x4 projInv = mat4r;
        Vc4x4 camInv = mat4r;
        Vc4x4 camInv2 = mat4r;

        Vc1 prob;
        iVc1 enable360;
        iVc1 interlace;
        iVc1 interlaceStage;
    };

    struct GeometryBlockUniform {
        GeometryUniformStruct geometryUniform = GeometryUniformStruct();
    };

    struct RayBlockUniform {
        SamplerUniformStruct samplerUniform = SamplerUniformStruct();
        CameraUniformStruct cameraUniform = CameraUniformStruct();
        MaterialUniformStruct materialUniform = MaterialUniformStruct();
    };





    struct MeshUniformStruct {
        GLint vertexAccessor = -1;
        GLint normalAccessor = -1;
        GLint texcoordAccessor = -1;
        GLint modifierAccessor = -1;

        glm::mat4 transform;
        glm::mat4 transformInv;

        GLint materialID = 0;
        GLint isIndexed = 0;
        GLint nodeCount = 1;
        GLint primitiveType = 0;

        GLint loadingOffset = 0;
        GLint storingOffset = 0;
        GLint _reserved0 = 1;
        GLint _reserved1 = 2;
    };


    struct VirtualAccessor {
        GLint offset = 0;
        GLint stride = 1;
        GLint components = 1;
        GLint type = 0; // 0 is float, 1 is uint, 2 is 16bit uint
    };


    struct VirtualMaterial {
        glm::vec4 diffuse = glm::vec4(0.0f);
        glm::vec4 specular = glm::vec4(0.0f);
        glm::vec4 transmission = glm::vec4(0.0f);
        glm::vec4 emissive = glm::vec4(0.0f);

        float ior = 1.0f;
        float roughness = 0.0001f;
        float alpharef = 0.0f;
        float unk0f = 0.0f;

        uint32_t diffusePart = 0;
        uint32_t specularPart = 0;
        uint32_t bumpPart = 0;
        uint32_t emissivePart = 0;

        int32_t flags = 0;
        int32_t alphafunc = 0;
        int32_t binding = 0;
        int32_t unk0i = 0;

        glm::ivec4 iModifiers0 = glm::ivec4(0);
    };

    struct GroupFoundResult {
        int nextResult = -1;
        float boxDistance = 100000.f;
        glm::ivec2 range;
    };

}