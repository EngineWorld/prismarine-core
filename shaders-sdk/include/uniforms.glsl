#ifndef _UNIFORMS_H
#define _UNIFORMS_H

#include "../include/structs.glsl"

#define RAY_BLOCK rayBlock. 
#define GEOMETRY_BLOCK geometryBlock.  

//#define RAY_BLOCK 
//#define GEOMETRY_BLOCK 

struct MaterialUniformStruct {
    iVc1 materialID;
    iVc1 _reserved;
    iVc1 time;
    iVc1 lightcount;
    Vc4 backgroundColor; // for skybox configure
    iVc4 iModifiers0;
    iVc4 iModifiers1;
    Vc4 fModifiers0;
    Vc4 fModifiers1;
    Vc4x4 transformModifier;
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
};

struct LightUniformStruct {
    Vc4 lightVector;
    Vc4 lightColor;
    Vc4 lightOffset;
    Vc4 lightAmbient;
};

struct GeometryUniformStruct {
    Vc4x4 transform;
    Vc4x4 transformInv;

    Vc4x4 gTransform;
    Vc4x4 gTransformInv;

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

struct OctreeUniformStruct {
    Vc4x4 project;
    Vc4x4 unproject;

    iVc1 maxDepth;
    iVc1 currentDepth;
    iVc1 nodeCount;
    iVc1 unk0;
};

struct CameraUniformStruct {
    Vc4x4 projInv;
    Vc4x4 camInv;
    Vc4x4 camInv2;

    Vc1 prob;
    iVc1 enable360;
    iVc1 interlace;
    iVc1 interlaceStage;
};

layout ( std430, binding = 12 ) readonly buffer LightUniform {
    LightUniformStruct lightNode[];
} lightUniform;

layout ( std430, binding = 13 ) readonly buffer RayBlockUniform {
    SamplerUniformStruct samplerUniform;
    CameraUniformStruct cameraUniform;
    MaterialUniformStruct materialUniform;
} rayBlock; 

layout ( std430, binding = 14 ) readonly buffer GeometryBlockUniform {
    GeometryUniformStruct geometryUniform;
} geometryBlock;

#endif
