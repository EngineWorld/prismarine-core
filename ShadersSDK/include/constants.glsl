
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


#ifndef _CONSTANTS_H
#define _CONSTANTS_H


// hardware or driver options
#define USE_INT64
#define INT64_MORTON
//#define EMULATE_BALLOT
//#define USE_ARB_CLOCK
//#define USE_ARB_PRECISION
//#define ENABLE_NVIDIA_INSTRUCTION_SET
//#define ENABLE_AMD_INSTRUCTION_SET
//#define ENABLE_INT16_LOADING // such as Neverball with GLushort
//#define ENABLE_UNSUPPOTED_FUNCTIONS


#ifdef INT64_MORTON
#define MORTONTYPE uint64_t
#else
#define MORTONTYPE uint
#endif

// ray tracing options
//#define EXPERIMENTAL_DOF
#define SUNLIGHT_CAUSTICS false
#define REFRACTION_SKIP_SUN
#define DIRECT_LIGHT_ENABLED

// sampling options
#define MOTION_BLUR
#define SAMPLES_LOCK 4 // required when using motion blur


// enable required OpenGL extensions
#ifdef ENABLE_AMD_INSTRUCTION_SET
#extension GL_AMD_gcn_shader : enable
#extension GL_AMD_gpu_shader_half_float : enable
#extension GL_AMD_gpu_shader_half_float2 : enable
#extension GL_AMD_gpu_shader_int16 : enable
#extension GL_AMD_shader_trinary_minmax : enable
#extension GL_AMD_texture_gather_bias_lod : enable
#endif

#ifdef ENABLE_NVIDIA_INSTRUCTION_SET
#extension GL_NV_gpu_shader5 : enable
#endif

#ifdef USE_INT64
#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_AMD_gpu_shader_int64 : enable
#endif

#ifdef USE_ARB_CLOCK
#extension GL_ARB_shader_clock : require
#endif

#ifdef USE_ARB_PRECISION
#extension GL_ARB_shader_precision : require
#endif

#ifndef EMULATE_BALLOT
#extension GL_ARB_shader_ballot : require
#extension GL_ARB_shader_group_vote : require
#endif


// System Constants
#define PZERO 0.0005f
#define COMPATIBLE_PRECISION 8388608.0

// Compute Shaders Definitions
#define WORK_SIZE 128
#define LOCAL_SIZE_LAYOUT layout ( local_size_x = WORK_SIZE ) in

// Math Constants
#define PHI 1.6180339887498948482
#define LONGEST -1
#define INFINITY 10000.0f
#define PI 3.1415926535897932384626422832795028841971
#define TWO_PI 6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476
#define E 2.7182818284590452353602874713526624977572

#endif
