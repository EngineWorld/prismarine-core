
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


#ifndef _VERTEX_H
#define _VERTEX_H

#include "../include/mathlib.glsl"

layout ( std430, binding = 10 ) restrict buffer GeomMaterialsSSBO {int mats[];};

layout (binding = 0) uniform sampler2D vertex_texture;
layout (binding = 1) uniform sampler2D normal_texture;
layout (binding = 2) uniform sampler2D texcoords_texture;
layout (binding = 3) uniform sampler2D modifiers_texture;

const ivec2 mit[3] = {ivec2(0,0), ivec2(1,0), ivec2(0,1)};

ivec2 mosaicIdc(in ivec2 mosaicCoord, in int idc){
    return mosaicCoord + mit[idc];
}

ivec2 gatherMosaic(in ivec2 uniformCoord){
    return ivec2(uniformCoord.x * 3 + uniformCoord.y % 3, uniformCoord.y);
}

vec4 fetchMosaic(in sampler2D vertices, in ivec2 mosaicCoord, in uint idc){
    return texelFetch(vertices, mosaicCoord + mit[idc], 0);
}

ivec2 getUniformCoord(in int indice){
    return ivec2(indice % 2047, indice / 2047);
}

ivec2 getUniformCoord(in uint indice){
    return ivec2(indice % 2047, indice / 2047);
}



vec2 dot2(in mat3x2 a, in mat3x2 b){
    return fma(a[0],b[0], fma(a[1],b[1], a[2]*b[2]));
    //mat2x3 at = transpose(a);
    //mat2x3 bt = transpose(b);
    //return vec2(dot(at[0], bt[0]), dot(at[1], bt[1]));
}




vec2 intersectTriangle2(in vec3 orig, in vec3 dir, inout ivec2 tri, inout vec4 UV, in bvec2 valid) {
    UV = vec4(0.f);

    vec2 t2 = vec2(INFINITY);

    valid = and2(valid, notEqual(tri, ivec2(LONGEST)));
    if (anyInvocationARB(any(valid))) {
        
        ivec2 tri0 = gatherMosaic(getUniformCoord(tri.x));
        ivec2 tri1 = gatherMosaic(getUniformCoord(tri.y));

        vec2 sz = 1.f / textureSize(vertex_texture, 0);
        vec2 hs = sz * 0.5f;
        vec2 ntri0 = fma(vec2(tri0), sz, hs);
        vec2 ntri1 = fma(vec2(tri1), sz, hs);

        mat3x2 v012x = transpose(mat2x3(
            textureGather(vertex_texture, ntri0, 0).wzx,
            textureGather(vertex_texture, ntri1, 0).wzx
        ));
        mat3x2 v012y = transpose(mat2x3(
            textureGather(vertex_texture, ntri0, 1).wzx,
            textureGather(vertex_texture, ntri1, 1).wzx
        ));
        mat3x2 v012z = transpose(mat2x3(
            textureGather(vertex_texture, ntri0, 2).wzx,
            textureGather(vertex_texture, ntri1, 2).wzx
        ));

/*
        mat3 ve0xyz = transpose(mat3(
            fetchMosaic(vertex_texture, tri0, 0).xyz, 
            fetchMosaic(vertex_texture, tri0, 1).xyz, 
            fetchMosaic(vertex_texture, tri0, 2).xyz
        ));

        mat3 ve1xyz = transpose(mat3(
            fetchMosaic(vertex_texture, tri1, 0).xyz, 
            fetchMosaic(vertex_texture, tri1, 1).xyz, 
            fetchMosaic(vertex_texture, tri1, 2).xyz
        ));

        mat3x2 v012x = transpose(mat2x3(ve0xyz[0], ve1xyz[0]));
        mat3x2 v012y = transpose(mat2x3(ve0xyz[1], ve1xyz[1]));
        mat3x2 v012z = transpose(mat2x3(ve0xyz[2], ve1xyz[2]));
*/

        mat3x2 e1 = mat3x2(v012x[1] - v012x[0], v012y[1] - v012y[0], v012z[1] - v012z[0]);
        mat3x2 e2 = mat3x2(v012x[2] - v012x[0], v012y[2] - v012y[0], v012z[2] - v012z[0]);
        mat3x2 dir2 = mat3x2(dir.xx, dir.yy, dir.zz);
        mat3x2 orig2 = mat3x2(orig.xx, orig.yy, orig.zz);

        mat3x2 pvec = mat3x2(
            fma(dir2[1], e2[2], - dir2[2] * e2[1]), 
            fma(dir2[2], e2[0], - dir2[0] * e2[2]), 
            fma(dir2[0], e2[1], - dir2[1] * e2[0])
        );

        vec2 det = dot2(pvec, e1);
        valid = and2(valid, greaterThan(abs(det), vec2(0.f)));
        if (anyInvocationARB(any(valid))) {
            vec2 invDev = 1.f / (max(abs(det), 0.000001f) * sign(det));
            mat3x2 tvec = mat3x2(
                orig2[0] - v012x[0],
                orig2[1] - v012y[0], 
                orig2[2] - v012z[0]
            );

            vec2 u = vec2(0.f);
            u = dot2(tvec, pvec) * invDev;
            valid = and2(valid, and2(greaterThanEqual(u, vec2(-0.00001f)), lessThan(u, vec2(1.00001f))));
            if (anyInvocationARB(any(valid))) {
                mat3x2 qvec = mat3x2(
                    fma(tvec[1], e1[2], -tvec[2] * e1[1]),
                    fma(tvec[2], e1[0], -tvec[0] * e1[2]),
                    fma(tvec[0], e1[1], -tvec[1] * e1[0])
                );

                vec2 v = vec2(0.f);
                v = dot2(dir2, qvec) * invDev;
                valid = and2(valid, and2(greaterThanEqual(v, vec2(-0.00001f)), lessThan(u+v, vec2(1.00001f))));
                if (anyInvocationARB(any(valid))) {
                    // distance
                    t2 = dot2(e2, qvec) * invDev;
                    valid = and2(valid, lessThan(t2, vec2(INFINITY - PZERO)));
                    valid = and2(valid, greaterThan(t2, vec2(0.0f - PZERO)));

                    UV = vec4(u, v);
                }
            }
        }
    }

    return mix(vec2(INFINITY), t2, valid);
}








// WARP optimized triangle intersection
float intersectTriangle(in vec3 orig, in vec3 dir, in int tri, inout vec2 UV, in bool valid) {
    // pre-invalidate
    float T = INFINITY;

    if (tri == LONGEST) valid = false;
    if (anyInvocationARB(valid)) {
        // fetch directly
        mat3 ve = mat3(
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 0).xyz, 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 1).xyz, 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 2).xyz
        );

        // init vars
        vec3 e1 = ve[1] - ve[0];
        vec3 e2 = ve[2] - ve[0];
        vec3 pvec = cross(dir, e2);
        float det = dot(e1, pvec);

        // invalidate culling
        if (abs(det) <= 0.0f) valid = false;
        if (anyInvocationARB(valid)) {
            // invalidate U
            float invDev = 1.f / (max(abs(det), 0.000001f) * sign(det));
            vec3 tvec = orig - ve[0];
            float u = dot(tvec, pvec) * invDev;
            if (u < -0.00001f || u > 1.00001f) valid = false;
            if (anyInvocationARB(valid)) {
                // invalidate V
                vec3 qvec = cross(tvec, e1);
                float v = dot(dir, qvec) * invDev;
                if (v < -0.00001f || (u+v) > 1.00001f) valid = false;
                if (anyInvocationARB(valid)) {
                    // resolve T
                    float t = dot(e2, qvec) * invDev;
                    if (greaterEqualF(t, 0.0f) && valid) {
                        T = t;
                        UV.xy = vec2(u, v);
                    }
                }
            }
        }
    }

    return T;
}


//#if defined(ENABLE_AMD_INSTRUCTION_SET) || defined(ENABLE_NVIDIA_INSTRUCTION_SET)
#if defined(ENABLE_NVIDIA_INSTRUCTION_SET)
#define INDEX16 uint16_t
#define M16(m, i) (m[i])
#else
#define INDEX16 uint
#define M16(m, i) (BFE_HW(m[i/2], int(16*(i&1)), 16))
#endif

#ifdef ENABLE_INT16_LOADING
#define INDICE_T INDEX16
#define PICK(m, i) M16(m, i)
#else
#define INDICE_T uint
#define PICK(m, i) m[i]
#endif


#endif
