
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


#ifndef _UNIFORMS_H
#define _UNIFORMS_H

#include "../include/structs.glsl"

#define RAY_BLOCK rayBlock. 
#define GEOMETRY_BLOCK geometryBlock.  

//#define RAY_BLOCK 
//#define GEOMETRY_BLOCK 

struct MaterialUniformStruct {
    int materialID;
    int _reserved;
    int time;
    int lightcount;
    vec4 backgroundColor; // for skybox configure
    ivec4 iModifiers0;
    ivec4 iModifiers1;
    vec4 fModifiers0;
    vec4 fModifiers1;
    mat4x4 transformModifier;
};

struct SamplerUniformStruct {
    vec2 sceneRes;
    int samplecount;
    int rayCount;
    int iteration;
    int phase;
    int hitCount; // planned
    int reserved0;
    int reserved1;
    int currentRayLimit;
};

struct LightUniformStruct {
    vec4 lightVector;
    vec4 lightColor;
    vec4 lightOffset;
    vec4 lightAmbient;
};

struct GeometryUniformStruct {
    mat4x4 transform;
    mat4x4 transformInv;

    int materialID;
    int triangleCount;
    int triangleOffset;
    int clearDepth;
};

struct CameraUniformStruct {
    mat4x4 projInv;
    mat4x4 camInv;
    mat4x4 prevCamInv;

    float prob;
    int enable360;
    int interlace;
    int interlaceStage;
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
