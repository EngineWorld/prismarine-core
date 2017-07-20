#pragma once

#include "includes.hpp"
#include "utils.hpp"

namespace Paper {
    
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
        Vc4 origin;
        Vc4 direct;
        Vc4 color;
        Vc4 final;
        iVc4 params;
        iVc1 idx;
        iVc1 prev;
        iVc1 bounce;
        iVc1 actived;
    };

    struct Hit {
        Vc4 normal;
        Vc4 tangent;
        Vc4 texcoord;
        Vc4 vcolor;
        Vc4 vmods;
        Vc1 dist;
        iVc1 triangle;
        iVc1 materialID;
        iVc1 shaded;
    };

    struct Texel {
        Vc4 coord;
        //Vc4 samplecolor;
        iVc4 EXT;
    };

    struct Leaf {
        bbox box;
        iVc4 pdata;
    };

    struct HlbvhNode {
        bbox box;
        iVc4 pdata;
        iVc4 leading;
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




    struct RandomUniformStruct {
        iVc1 time;
        Vc1 _reserved0;
        Vc1 _reserved1;
        Vc1 _reserved2;
    };

    struct MinmaxUniformStruct {
        iVc1 heap;
        Vc1 prec;
        Vc1 _reserved0;
        Vc1 _reserved1;
    };

    struct MaterialUniformStruct {
        Vc1 materialID;
        Vc1 f_shadows;
        Vc1 f_reflections;
        Vc1 lightcount;
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
        iVc1 maxSamples;
        iVc1 currentSample;
        iVc1 maxFilters;
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
        Vc4x4 texmatrix = mat4r;
        Vc4 colormod = vec4r;

        Vc1 offset = 0.0f;
        iVc1 materialID = 0;
        iVc1 triangleCount = 1;
        iVc1 triangleOffset = 0;

        iVc1 unindexed = 0;
        iVc1 loadOffset = 0;
        iVc1 NB_mode;
        iVc1 clearDepth = 0;
    };

    struct OctreeUniformStruct {
        Vc4x4 project = mat4r;
        Vc4x4 unproject = mat4r;
        Vc4x4 transform = mat4r;
        Vc4x4 transformInv = mat4r;

        iVc1 maxDepth;
        iVc1 currentDepth;
        iVc1 nodeCount;
        iVc1 unk0;
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

    struct AttributeUniformStruct {
        iVc1 vertexOffset = 0;
        iVc1 normalOffset = 0;
        iVc1 texcoordOffset = 0;
        iVc1 lightcoordOffset = 0;

        iVc1 colorOffset = 0;
        iVc1 stride = 3;
        iVc1 mode = 0;
        iVc1 colorFormat = 0;

        iVc1 haveColor = 0;
        iVc1 haveNormal = 0;
        iVc1 haveTexcoord = 0;
        iVc1 haveLightcoord = 0;

        // for future shaders
        iVc4 iModifiers0 = ivec4r;
        iVc4 iModifiers1 = ivec4r;
        Vc4 fModifiers0 = vec4r;
        Vc4 fModifiers1 = vec4r;
    };

    struct GeometryBlockUniform {
        AttributeUniformStruct attributeUniform = AttributeUniformStruct();
        GeometryUniformStruct geometryUniform = GeometryUniformStruct();
        OctreeUniformStruct octreeUniform = OctreeUniformStruct();
        MinmaxUniformStruct minmaxUniform = MinmaxUniformStruct();
    };

    struct RayBlockUniform {
        SamplerUniformStruct samplerUniform = SamplerUniformStruct();
        CameraUniformStruct cameraUniform = CameraUniformStruct();
        MaterialUniformStruct materialUniform = MaterialUniformStruct();
        RandomUniformStruct randomUniform = RandomUniformStruct();
    };


}