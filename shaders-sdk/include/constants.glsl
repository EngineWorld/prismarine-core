#ifndef _CONSTANTS_H
#define _CONSTANTS_H

/* Water and optics settings */
#define PHI 1.6180339887498948482
#define LONGEST 0xFFFFFFFF
#define INFINITY 100000.0f
#define GAMMA 1.0f
#define PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define E 2.7182818284590452353602874713526624977572f

#define DIRECT_LIGHT
#define PZERO  2.e-4
#define NZERO -PZERO

//#define ENABLE_AMD_INSTRUCTION_SET
//#define ENABLE_INT16_LOADING // such as Neverball

#define ENABLE_NVIDIA_INSTRUCTION_SET
#define COMPATIBLE_PRECISION 8388608.0

#define SUNLIGHT_CAUSTICS false
#define REFRACTION_SKIP_SUN

//#define FREEIMAGE_STYLE
//#define RT_OPTIMIZED
//#define USE_EMULATED
//#define INVERT_TEXCOORD
//#define CAD_SYSTEM_CUBEMAP
//#define SYSTEM_180_CUBEMAP
#define MOTION_BLUR
#define USE_INT64
//#define INT64_MORTON

//#define CULLING

#ifdef ENABLE_NVIDIA_INSTRUCTION_SET
#extension GL_NV_shader_atomic_float : require
#endif

#ifdef USE_INT64
#extension GL_ARB_gpu_shader_int64 : require
#endif

#define WORK_SIZE 128
//#define LOCAL_SIZE_LAYOUT layout ( local_size_variable ) in;
//#extension GL_ARB_compute_variable_group_size : require
#define LOCAL_SIZE_LAYOUT layout ( local_size_x = WORK_SIZE ) in

#endif
