#ifndef _VERTEX_H
#define _VERTEX_H

#include "../include/STOmath.glsl"

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
    if (comp == 0) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 0); else 
    if (comp == 1) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 1); else 
    if (comp == 2) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 2); else 
    if (comp == 3) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 3); else 
    return vec4(0.f);
}


vec4 fetchMosaic(in sampler2D vertices, in ivec2 mosaicCoord, in uint idc){
    return texelFetch(vertices, mosaicCoord + mit[idc], 0);
}

ivec2 getUniformCoord(in int indice){
    return ivec2(indice % 1023, indice / 1023);
}

ivec2 getUniformCoord(in uint indice){
    return ivec2(indice % 1023, indice / 1023);
}


float intersectTriangle4(in vec3 orig, in vec3 dir, in ivec4 tri, inout vec2 UV, inout int triID) {
    UV = vec2(0.f);
    triID = LONGEST;
    float t = INFINITY;

    // check valid triangles
    bvec4 valid = bvec4(notEqual(tri, ivec4(LONGEST)));

    // storing triangles in vector components
    mat3x4 v012x = transpose(mat4x3(
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 0).wzx, // triangle 0, verts 0, 1, 2
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 0).wzx, // triangle 1
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.z)), 0).wzx, // triangle 2
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.w)), 0).wzx  // triangle 3
    ));
    mat3x4 v012y = transpose(mat4x3(
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 1).wzx,
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 1).wzx,
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.z)), 1).wzx,
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.w)), 1).wzx
    ));
    mat3x4 v012z = transpose(mat4x3(
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.x)), 2).wzx,
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.y)), 2).wzx,
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.z)), 2).wzx,
        gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri.w)), 2).wzx
    ));

    // gather e1, e2
    vec4 e1x = v012x[1] - v012x[0];
    vec4 e1y = v012y[1] - v012y[0];
    vec4 e1z = v012z[1] - v012z[0];
    vec4 e2x = v012x[2] - v012x[0];
    vec4 e2y = v012y[2] - v012y[0];
    vec4 e2z = v012z[2] - v012z[0];

    // get ray dir
    vec4 dir4x = dir.xxxx;
    vec4 dir4y = dir.yyyy;
    vec4 dir4z = dir.zzzz;

    // division (det)
    vec4 pvecx = fma(dir4y, e2z, -dir4z*e2y);
    vec4 pvecy = fma(dir4z, e2x, -dir4x*e2z);
    vec4 pvecz = fma(dir4x, e2y, -dir4y*e2x);
    vec4 divisor = fma(pvecx, e1x, fma(pvecy, e1y, pvecz*e1z));
    valid = and(valid, greaterThan(abs(divisor), vec4(0.f)));
    if (all(not(valid))) return t;
    vec4 invDivisor = vec4(1) / divisor;

    // get ray orig
    vec4 orig4x = orig.xxxx;
    vec4 orig4y = orig.yyyy;
    vec4 orig4z = orig.zzzz;

    // U
    vec4 tvecx = orig4x - v012x[0];
    vec4 tvecy = orig4y - v012y[0];
    vec4 tvecz = orig4z - v012z[0];
    vec4 u4;
    u4 = fma(tvecx, pvecx, fma(tvecy, pvecy, tvecz*pvecz));
    u4 = u4 * invDivisor;
    valid = and(valid, and(greaterThanEqual(u4, vec4(0.f)), lessThan(u4, vec4(1.f))));
    if (all(not(valid))) return t;

    // V
    vec4 qvecx = fma(tvecy, e1z, -tvecz*e1y);
    vec4 qvecy = fma(tvecz, e1x, -tvecx*e1z);
    vec4 qvecz = fma(tvecx, e1y, -tvecy*e1x);
    vec4 v4;
    v4 = fma(dir4x, qvecx, fma(dir4y, qvecy, dir4z*qvecz));
    v4 = v4 * invDivisor;
    valid = and(valid, and(greaterThanEqual(v4, vec4(0.f)), lessThan(u4+v4, vec4(1.f))));
    if (all(not(valid))) return t;

    // distance
    vec4 t4;
    t4 = fma(e2x, qvecx, fma(e2y, qvecy, e2z*qvecz));
    t4 = t4 * invDivisor;
    valid = and(valid, lessThan(t4, vec4(INFINITY - PZERO)));
    if (all(not(valid))) return t;

    // ordered resulting
    if ((equalF(t4.x, t) ? tri.x < triID : lessEqualF(t4.x, t)) && greaterEqualF(t4.x, 0))
    if (valid.x) { t = t4.x; triID = tri.x; UV.xy = vec2(u4.x, v4.x); }

    if ((equalF(t4.y, t) ? tri.y < triID : lessEqualF(t4.y, t)) && greaterEqualF(t4.y, 0))
    if (valid.y) { t = t4.y; triID = tri.y; UV.xy = vec2(u4.y, v4.y); }

    if ((equalF(t4.z, t) ? tri.z < triID : lessEqualF(t4.z, t)) && greaterEqualF(t4.z, 0))
    if (valid.z) { t = t4.z; triID = tri.z; UV.xy = vec2(u4.z, v4.z); }

    if ((equalF(t4.w, t) ? tri.w < triID : lessEqualF(t4.w, t)) && greaterEqualF(t4.w, 0))
    if (valid.w) { t = t4.w; triID = tri.w; UV.xy = vec2(u4.w, v4.w); }

    return t;
}

// WARP optimized triangle intersection
float intersectTriangle(in vec3 orig, in vec3 dir, in int tri, inout vec2 UV, in bool valid) {
    // pre-invalidate
    if (tri == LONGEST) valid = false;
    if (allInvocations(!valid)) return INFINITY;

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
    if (allInvocations(!valid)) return INFINITY;

    // invalidate U
    vec3 tvec = orig - ve[0];
    float u = dot(tvec, pvec) / det;
    if (u < 0.f || u > 1.0f) valid = false;
    if (allInvocations(!valid)) return INFINITY;

    // invalidate V
    vec3 qvec = cross(tvec, e1);
    float v = dot(dir, qvec) / det;
    if (v < 0.f || (u+v) > 1.0f) valid = false;
    if (allInvocations(!valid)) return INFINITY;

    // resolve T
    float t = dot(e2, qvec) / det;
    UV.xy = vec2(u, v);
    return (lessF(t, 0.0f) || !valid) ? INFINITY : t;
}


#endif
