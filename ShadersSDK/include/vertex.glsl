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

vec4 gatherMosaicCompDyn(in sampler2D vertices, in ivec2 mosaicCoord, const uint comp){
    vec4 components = vec4(0.0f);
    vec2 iTexSize = 1.0f / textureSize(vertices, 0);
    if (comp == 0) components = textureGather(vertices, (vec2(mosaicCoord) + 0.5f) * iTexSize, 0); else 
    if (comp == 1) components = textureGather(vertices, (vec2(mosaicCoord) + 0.5f) * iTexSize, 1); else 
    if (comp == 2) components = textureGather(vertices, (vec2(mosaicCoord) + 0.5f) * iTexSize, 2); else 
    if (comp == 3) components = textureGather(vertices, (vec2(mosaicCoord) + 0.5f) * iTexSize, 3); else 
    return components;
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



vec2 intersectTriangle2(in vec3 orig, in vec3 dir, inout ivec2 tri, inout vec4 UV, in bvec2 valid) {
    UV = vec4(0.f);

    vec2 t2 = vec2(INFINITY);

    valid = and2(valid, notEqual(tri, ivec2(LONGEST)));
    if (anyInvocation(any(valid))) {

/*
        // does not work (broken OpenGL support)
        mat3x2 v012x = transpose(mat2x3(
            gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 0).wzx, // triangle 0, verts 0, 1, 2
            gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 0).wzx  // triangle 1
        ));
        mat3x2 v012y = transpose(mat2x3(
            gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 1).wzx,
            gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 1).wzx
        ));
        mat3x2 v012z = transpose(mat2x3(
            gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 2).wzx,
            gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 2).wzx
        ));
*/

        mat3 ve0 = mat3(
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 0).xyz, 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 1).xyz, 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 2).xyz
        );

        mat3 ve1 = mat3(
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 0).xyz, 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 1).xyz, 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 2).xyz
        );

        mat3x2 v012x = mat3x2(
            vec2(ve0[0].x, ve1[0].x),
            vec2(ve0[1].x, ve1[1].x),
            vec2(ve0[2].x, ve1[2].x)
        );

        mat3x2 v012y = mat3x2(
            vec2(ve0[0].y, ve1[0].y),
            vec2(ve0[1].y, ve1[1].y),
            vec2(ve0[2].y, ve1[2].y)
        );

        mat3x2 v012z = mat3x2(
            vec2(ve0[0].z, ve1[0].z),
            vec2(ve0[1].z, ve1[1].z),
            vec2(ve0[2].z, ve1[2].z)
        );

        mat3x2 e1 = mat3x2(v012x[1] - v012x[0], v012y[1] - v012y[0], v012z[1] - v012z[0]);
        mat3x2 e2 = mat3x2(v012x[2] - v012x[0], v012y[2] - v012y[0], v012z[2] - v012z[0]);
        mat3x2 dir2 = mat3x2(dir.xx, dir.yy, dir.zz);
        mat3x2 orig2 = mat3x2(orig.xx, orig.yy, orig.zz);

        mat3x2 pvec = mat3x2(
            dir2[1] * e2[2] - dir2[2] * e2[1], 
            dir2[2] * e2[0] - dir2[0] * e2[2], 
            dir2[0] * e2[1] - dir2[1] * e2[0]
        );

        vec2 det = pvec[0]*e1[0] + pvec[1]*e1[1] + pvec[2]*e1[2];
        valid = and2(valid, greaterThan(abs(det), vec2(0.f)));
        if (anyInvocation(any(valid))) {
            vec2 invDev = 1.f / (max(abs(det), 0.000001f) * sign(det));
            mat3x2 tvec = mat3x2(
                orig2[0] - v012x[0],
                orig2[1] - v012y[0], 
                orig2[2] - v012z[0]
            );

            vec2 u = vec2(0.f);
            u = (tvec[0]*pvec[0] + tvec[1]*pvec[1] + tvec[2]*pvec[2]) * invDev;
            valid = and2(valid, and2(greaterThanEqual(u, vec2(0.f)), lessThan(u, vec2(1.f))));
            if (anyInvocation(any(valid))) {
                mat3x2 qvec = mat3x2(
                    tvec[1] * e1[2] - tvec[2] * e1[1],
                    tvec[2] * e1[0] - tvec[0] * e1[2],
                    tvec[0] * e1[1] - tvec[1] * e1[0]
                );

                vec2 v = vec2(0.f);
                v = (dir2[0]*qvec[0] + dir2[1]*qvec[1] + dir2[2]*qvec[2]) * invDev;
                valid = and2(valid, and2(greaterThanEqual(v, vec2(0.f)), lessThan(u+v, vec2(1.f))));
                if (anyInvocation(any(valid))) {
                    // distance
                    t2 = (e2[0]*qvec[0] + e2[1]*qvec[1] + e2[2]*qvec[2]) * invDev;
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
    if (anyInvocation(valid)) {
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
        if (anyInvocation(valid)) {
            // invalidate U
            float invDev = 1.f / (max(abs(det), 0.000001f) * sign(det));
            vec3 tvec = orig - ve[0];
            float u = dot(tvec, pvec) * invDev;
            if (u < 0.f || u > 1.0f) valid = false;
            if (anyInvocation(valid)) {
                // invalidate V
                vec3 qvec = cross(tvec, e1);
                float v = dot(dir, qvec) * invDev;
                if (v < 0.f || (u+v) > 1.0f) valid = false;
                if (anyInvocation(valid)) {
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


#if defined(ENABLE_AMD_INSTRUCTION_SET) || defined(ENABLE_NVIDIA_INSTRUCTION_SET)
#define INDEX16 uint16_t
#define M16(m, i) (m[i])
#else
#define INDEX16 uint
#define M16(m, i) (BFE(m[i/2], int(16*(i%2)), 16))
#endif

#ifdef ENABLE_INT16_LOADING
#define INDICE_T INDEX16
#define PICK(m, i) M16(m, i)
#else
#define INDICE_T uint
#define PICK(m, i) m[i]
#endif


#endif
