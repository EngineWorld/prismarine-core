#ifndef _RAYS_H
#define _RAYS_H

#include "../include/STOmath.glsl"
#include "../include/morton.glsl"

layout ( std430, binding = 0 )  buffer RaysSSBO { Ray nodes[]; } rayBuf;
layout ( std430, binding = 1 )  buffer HitsSSBO { Hit nodes[]; } hitBuf;
layout ( std430, binding = 2 )  buffer TexelsSSBO { Texel nodes[]; } texelBuf;
layout ( std430, binding = 6 ) readonly buffer ActivedIndicesSSBO { int indc[]; } activedBuf;
layout ( std430, binding = 7 )  buffer CollectedActivesSSBO { int indc[]; } collBuf;
layout ( std430, binding = 8 )  buffer FreedomIndicesSSBO { int indc[]; } freedBuf;
layout ( std430, binding = 14 ) readonly buffer AvailablesIndicesSSBO { int indc[]; } availBuf;
layout ( std430, binding = 20 )  buffer CounterBlock { 
    int At;
    int Rt;
    int Qt;
    int Ut;
    int Ct;
} arcounter;
layout ( std430, binding = 21 )  buffer ColorChainBlock { ColorChain chains[]; } chBuf;

/*
const uint At = 0;
const uint Rt = 1;
const uint Qt = 2;
const uint Ut = 3;
const uint Ct = 4;
*/

void _collect(inout Ray ray) {
    const vec4 color = max(ray.final, vec4(0.f));
    const float amplitude = mlength(color.xyz);
    if (amplitude >= 0.00001f) {
        int idx = atomicAdd(arcounter.Ct, 1);
        int prev = atomicExchange(texelBuf.nodes[ray.texel].EXT.y, idx);
        ColorChain ch = chBuf.chains[idx];
        ch.color = ray.final;
        ch.cdata.x = prev;
        chBuf.chains[idx] = ch;
    }
    ray.final.xyzw = vec4(0.0f);

    /*
    const vec4 color = max(ray.final, vec4(0.f));
    const float amplitude = mlength(color.xyz);
    if (amplitude >= 0.00001f) {
#ifdef ENABLE_NVIDIA_INSTRUCTION_SET
        atomicAdd(texelBuf.nodes[ray.texel].samplecolor.x, color.x);
        atomicAdd(texelBuf.nodes[ray.texel].samplecolor.y, color.y);
        atomicAdd(texelBuf.nodes[ray.texel].samplecolor.z, color.z);
#else
        const ivec3 gcol = ivec3(dvec3(color.xyz) * COMPATIBLE_PRECISION);
        atomicAdd(texelBuf.nodes[ray.texel].samplecolor.x, gcol.x);
        atomicAdd(texelBuf.nodes[ray.texel].samplecolor.y, gcol.y);
        atomicAdd(texelBuf.nodes[ray.texel].samplecolor.z, gcol.z);
#endif
    }
    ray.final.xyzw = vec4(0.0f);
    */
}

void storeHit(in int hitIndex, inout Hit hit) {
    if (hitIndex == -1 || hitIndex == LONGEST || hitIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        return;
    }
    hitBuf.nodes[hitIndex] = hit;
}

void storeHit(inout Ray ray, inout Hit hit) {
    storeHit(ray.idx, hit);
}

int storeRay(in int rayIndex, inout Ray ray) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        return -1;
    }
    _collect(ray);

    int actived = -1;
    if (ray.actived == 1) {
        const int act = atomicAdd(arcounter.At, 1);
        collBuf.indc[act] = rayIndex; actived = act;
    } else { // if not actived, why need?
        const int freed = atomicAdd(arcounter.Qt, 1);
        freedBuf.indc[freed] = rayIndex;
    }

    ray.idx = rayIndex;
    rayBuf.nodes[rayIndex] = ray;
    return actived;
}

void storeRay(inout Ray ray) {
    storeRay(ray.idx, ray);
}

int createRayStrict(inout Ray original, in int idx, in int rayIndex) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        return rayIndex;
    }

    Ray ray = original;
    ray.idx = rayIndex;
    ray.bounce = ray.bounce;
    ray.texel = idx;
    ray.actived = ray.actived;

    Hit hit;
    if (original.idx != LONGEST) {
        hit = hitBuf.nodes[original.idx];
    } else {
        hit.normal = vec4(0.0f);
        hit.tangent = vec4(0.0f);
        hit.vmods = vec4(0.0f);
        hit.triangle = LONGEST;
        hit.materialID = LONGEST;
    }
    hit.shaded = 1;

    hitBuf.nodes[rayIndex] = hit;
    rayBuf.nodes[rayIndex] = ray;

    // if not active, does not use and free for nexts
    if(ray.actived == 1) {
        const int act = atomicAdd(arcounter.At, 1);
        collBuf.indc[act] = rayIndex;
    } else {
        const int freed = atomicAdd(arcounter.Qt, 1);
        freedBuf.indc[freed] = rayIndex;
    }
    return rayIndex;
}

int createRayStrict(inout Ray original, in int rayIndex) {
    return createRayStrict(original, original.texel, rayIndex);
}



int createRay(inout Ray original, in int idx) {
    _collect(original);
    if (
        original.actived < 1 || 
        original.bounce < 0 || 
        mlength(original.color.xyz) < 0.00001f
    ) return -1; 
    
    atomicMax(arcounter.Ut, 0); // prevent most decreasing
    const int freed = atomicAdd(arcounter.Ut, -1)-1;
    atomicMax(arcounter.Ut, 0); // prevent most decreasing
    int rayIndex = 0;
    if (freed >= 0 && availBuf.indc[freed] != 0xFFFFFFFF) {
        rayIndex = availBuf.indc[freed];
        //availBuf.indc[freed] = 0xFFFFFFFF;
    } else {
        rayIndex = atomicAdd(arcounter.Rt, 1);
    }
    return createRayStrict(original, idx, rayIndex);
}

int createRayIdx(inout Ray original, in int idx, in int rayIndex) {
    _collect(original);
    if (
        original.actived < 1 || 
        original.bounce < 0 || 
        mlength(original.color.xyz) < 0.00001f
    ) return -1; 
    
    atomicMax(arcounter.Rt, rayIndex+1);
    return createRayStrict(original, idx, rayIndex);
}




int createRay(in Ray original) {
    return createRay(original, original.texel);
}

int createRay(in int idx) {
    Ray newRay;
    return createRay(newRay, idx);
}

Ray fetchRayDirect(in int texel) {
    return rayBuf.nodes[texel];
}

Hit fetchHitDirect(in int texel) {
    return hitBuf.nodes[texel];
}

Hit fetchHit(in Ray ray){
    return hitBuf.nodes[ray.idx];
}



uvec3 _roundly(in vec3 o){
    return clamp(
        uvec3(floor(clamp(o, vec3(0.00001f), vec3(0.99999f)) * 1024.0f)), 
        uvec3(0), uvec3(1023));
}

uint quantizeRay(in Ray ray, in vec3 mn, in vec3 mx){
    vec3 origin = (ray.origin.xyz - mn) / (mx - mn);
    vec3 direct = fma(normalize(ray.direct.xyz),vec3(0.5f),vec3(0.5f));
    return encodeMorton3_64(_roundly(origin), _roundly(direct));
    //return encodeMorton3_64(_roundly(direct), _roundly(origin));
}



#endif
