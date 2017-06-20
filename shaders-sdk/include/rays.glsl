#ifndef _RAYS_H
#define _RAYS_H

#include "../include/STOmath.glsl"

layout ( std430, binding = 0 ) buffer RaysSSBO { Ray rays[]; };
layout ( std430, binding = 1 ) buffer HitsSSBO { Hit hits[]; };
layout ( std430, binding = 2 ) buffer TexelsSSBO { Texel texelInfo[]; };
layout ( std430, binding = 6 ) readonly buffer ActivedIndicesSSBO { int actived[]; };
layout ( std430, binding = 7 ) buffer CollectedActivesSSBO { int qrays[]; };
layout ( std430, binding = 8 ) buffer FreedomIndicesSSBO { int freedoms[]; };
layout ( std430, binding = 14 ) readonly buffer AvailablesIndicesSSBO { int availables[]; };
layout ( std430, binding = 20 ) buffer CounterBlock { int arcounter[4]; };

const uint At = 0;
const uint Rt = 1;
const uint Qt = 2;
const uint Ut = 3;

void _collect(inout Ray ray) {
    const vec4 color = max(ray.final, vec4(0.f));
    const float amplitude = mlength(color.xyz);
    if (lessEqualF(amplitude, 0.f) || greaterF(amplitude, 1000.0f)) {
        ray.final.xyzw = vec4(0.0f);
        return;
    }
#ifdef ENABLE_NVIDIA_INSTRUCTION_SET
    atomicAdd(texelInfo[ray.texel].samplecolor.x, color.x);
    atomicAdd(texelInfo[ray.texel].samplecolor.y, color.y);
    atomicAdd(texelInfo[ray.texel].samplecolor.z, color.z);
#else
    const ivec3 gcol = ivec3(dvec3(color.xyz) * COMPATIBLE_PRECISION);
    atomicAdd(texelInfo[ray.texel].samplecolor.x, gcol.x);
    atomicAdd(texelInfo[ray.texel].samplecolor.y, gcol.y);
    atomicAdd(texelInfo[ray.texel].samplecolor.z, gcol.z);
#endif
    ray.final.xyzw = vec4(0.0f);
}

void storeHit(in int hitIndex, inout Hit hit) {
    if (hitIndex == -1 || hitIndex == LONGEST || hitIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        return;
    }
    hits[hitIndex] = hit;
}

void storeHit(inout Ray ray, inout Hit hit) {
    storeHit(ray.idx, hit);
}

void storeRay(in int rayIndex, inout Ray ray) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        return;
    }
    _collect(ray);

    if (ray.actived == 1) {
        const int act = atomicAdd(arcounter[At], 1);
        qrays[act] = rayIndex;
    } else { // if not actived, why need?
        const int freed = atomicAdd(arcounter[Qt], 1);
        freedoms[freed] = rayIndex;
    }

    ray.idx = rayIndex;
    rays[rayIndex] = ray;
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
    ray.texel = idx;

    Hit hit;
    if (original.idx != LONGEST) {
        hit = hits[original.idx];
    } else {
        hit.triangle = LONGEST;
        hit.normal = vec4(0.0f);
        hit.tangent = vec4(0.0f);
        hit.materialID = LONGEST;
        hit.vmods = vec4(0.0f);
    }
    hit.shaded = 1;

    hits[rayIndex] = hit;
    rays[rayIndex] = ray;

    // if not active, does not use and free for nexts
    if(ray.actived == 1) {
        const int act = atomicAdd(arcounter[At], 1);
        qrays[act] = rayIndex;
    } else {
        const int freed = atomicAdd(arcounter[Qt], 1);
        freedoms[freed] = rayIndex;
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
        lessEqualF(mlength(original.color.xyz), 0.f)
    ) return -1; 
    
    atomicMax(arcounter[Ut], 0); // prevent most decreasing
    const int freed = atomicAdd(arcounter[Ut], -1)-1;
    atomicMax(arcounter[Ut], 0); // prevent most decreasing
    int rayIndex = 0;
    if (freed >= 0 && availables[freed] != 0xFFFFFFFF) {
        rayIndex = availables[freed];
        //availables[freed] = 0xFFFFFFFF;
    } else {
        rayIndex = atomicAdd(arcounter[Rt], 1);
    }
    return createRayStrict(original, idx, rayIndex);
}

int createRayIdx(inout Ray original, in int idx, in int rayIndex) {
    _collect(original);
    if (
        original.actived < 1 || 
        original.bounce < 0 || 
        lessEqualF(mlength(original.color.xyz), 0.f)
    ) return -1; 
    
    atomicMax(arcounter[Rt], rayIndex+1);
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
    return rays[texel];
}

Hit fetchHitDirect(in int texel) {
    return hits[texel];
}

Hit fetchHit(in Ray ray){
    return hits[ray.idx];
}

#endif
