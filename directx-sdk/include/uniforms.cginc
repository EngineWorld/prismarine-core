#ifndef _UNIFORMS_H
#define _UNIFORMS_H

#include "../include/structs.cginc"

struct LightUniformStruct {
    Vc4 lightVector;
    Vc4 lightColor;
    Vc4 lightOffset;
    Vc4 lightAmbient;
};

struct RayBlockUniform {
    Vc2 sceneRes;
    iVc1 samplecount;
    iVc1 rayCount;
    iVc1 iteration;
    iVc1 phase;
    iVc1 maxSamples;
    iVc1 currentSample;
    iVc1 maxFilters;
    iVc1 currentRayLimit;

    Vc4x4 projInv;
    Vc4x4 camInv;
    Vc4x4 camInv2;

    Vc1 prob;
    iVc1 enable360;
    iVc1 interlace;
    iVc1 interlaceStage;

    Vc1 materialID;
    Vc1 f_shadows;
    Vc1 f_reflections;
    Vc1 lightcount;
    Vc4 backgroundColor; // for skybox configure
    iVc4 iModifiers0;
    iVc4 iModifiers1;
    Vc4 fModifiers0;
    Vc4 fModifiers1;
    Vc4x4 transformModifier;

    uint time;
    Vc1 _reserved0;
    Vc1 _reserved1;
    Vc1 _reserved2;
};

struct GeometryBlockUniform {
    iVc1 vertexOffset;
    iVc1 normalOffset;
    iVc1 texcoordOffset;
    iVc1 lightcoordOffset;

    iVc1 colorOffset;
    iVc1 stride;
    iVc1 mode;
    iVc1 colorFormat;

    iVc1 haveColor;
    iVc1 haveNormal;
    iVc1 haveTexcoord;
    iVc1 haveLightcoord;

    iVc4 iModifiers0;
    iVc4 iModifiers1;
    Vc4 fModifiers0;
    Vc4 fModifiers1;

    Vc4x4 transform;
    Vc4x4 transformInv;
    Vc4x4 texmatrix;
    Vc4 colormod;

    Vc1 offset;
    iVc1 materialID;
    iVc1 triangleCount;
    iVc1 triangleOffset;

    iVc1 unindexed;
    iVc1 loadOffset;
    iVc1 NB_mode;
    iVc1 clearDepth;
};

RWStructuredBuffer<LightUniformStruct> lightNode : register(u12);
RWStructuredBuffer<RayBlockUniform> rayBlock : register(u13);
RWStructuredBuffer<GeometryBlockUniform> geometryBlock : register(u14);

#endif
