#ifndef _RAYS_H
#define _RAYS_H

#include "../include/STOmath.glsl"
#include "../include/morton.glsl"

/*
layout ( std430, binding = 0 ) restrict buffer RaysSSBO { Ray nodes[]; } rayBuf;
layout ( std430, binding = 1 ) restrict buffer HitsSSBO { Hit nodes[]; } hitBuf;
layout ( std430, binding = 2 ) restrict buffer TexelsSSBO { Texel nodes[]; } texelBuf;
layout ( std430, binding = 6 ) readonly buffer ActivedIndicesSSBO { int indc[]; } activedBuf;
layout ( std430, binding = 7 ) restrict buffer CollectedActivesSSBO { int indc[]; } collBuf;
layout ( std430, binding = 8 ) restrict buffer FreedomIndicesSSBO { int indc[]; } freedBuf;
layout ( std430, binding = 14 ) readonly buffer AvailablesIndicesSSBO { int indc[]; } availBuf;
layout ( std430, binding = 20 ) restrict buffer CounterBlock { 
    int At;
    int Rt;
    int Qt;
    int Ut;
    int Ct;
} arcounter;
layout ( std430, binding = 21 ) restrict buffer ColorChainBlock { ColorChain chains[]; } chBuf;
*/



layout ( std430, binding = 0 ) restrict buffer RaysSSBO { Ray nodes[]; } rayBuf;
layout ( std430, binding = 1 ) restrict buffer HitsSSBO { Hit nodes[]; } hitBuf;
layout ( std430, binding = 2 ) restrict buffer TexelsSSBO { Texel nodes[]; } texelBuf;
layout ( std430, binding = 3 ) restrict buffer ColorChainBlock { ColorChain chains[]; } chBuf;

// current list
layout ( std430, binding = 4 ) readonly buffer ActivedIndicesSSBO { int indc[]; } activedBuf;
layout ( std430, binding = 5 ) readonly buffer AvailablesIndicesSSBO { int indc[]; } availBuf;

// new list
layout ( std430, binding = 6 ) restrict buffer CollectedActivesSSBO { int indc[]; } collBuf;
layout ( std430, binding = 7 ) restrict buffer FreedomIndicesSSBO { int indc[]; } freedBuf;

// counters
layout ( std430, binding = 8 ) restrict buffer CounterBlock { 
    int At; // new list collection counter
    int Rt; // ray list counter
    int Qt; // next available ptr 
    int Ut; // free list counter
    int Ct; // color chain list counter
} arcounter;




void _collect(inout Ray ray) {
    vec4 color = max(ray.final, vec4(0.f));
    float amplitude = mlength(color.xyz);
    if (amplitude >= 0.00001f) {
        int idx = atomicAdd(arcounter.Ct, 1);
        int prev = atomicExchange(texelBuf.nodes[ray.texel].EXT.y, idx);
        ColorChain ch = chBuf.chains[idx];
        ch.color = ray.final;
        ch.cdata.x = prev;
        chBuf.chains[idx] = ch;
    }
    ray.final.xyzw = vec4(0.0f);
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


int addRayToList(in Ray ray){
    int rayIndex = ray.idx;
    int actived = -1;
    if (ray.actived == 1) {
        int act = atomicAdd(arcounter.At, 1);
        collBuf.indc[act] = rayIndex; actived = act;
    } else { // if not actived, why need?
        int freed = atomicAdd(arcounter.Qt, 1);
        freedBuf.indc[freed] = rayIndex;
    }
    return actived;
}

int addRayToList(in Ray ray, in int act){
    int rayIndex = ray.idx;
    int actived = -1;
    if (ray.actived == 1) {
        collBuf.indc[act] = rayIndex; actived = act;
    }
    return actived;
}


void storeRay(in int rayIndex, inout Ray ray) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        ray.actived = 0;
        return;
    }
    _collect(ray);

    ray.idx = rayIndex;
    rayBuf.nodes[rayIndex] = ray;
}

void storeRay(inout Ray ray) {
    storeRay(ray.idx, ray);
}

int createRayStrict(inout Ray original, in int idx, in int rayIndex) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        return rayIndex;
    }

    bool invalidRay = 
        original.actived < 1 || 
        original.bounce <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    //if (allInvocationsARB(invalidRay)) {
    if (invalidRay) {
        return rayIndex; 
    }

    Ray ray = original;
    ray.bounce -= 1;
    ray.idx = rayIndex;
    ray.texel = idx;

    // mark as unusual
    if (invalidRay) {
        ray.actived = 0;
    }

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

    addRayToList(ray);
    return rayIndex;
}

int createRayStrict(inout Ray original, in int rayIndex) {
    return createRayStrict(original, original.texel, rayIndex);
}



int createRay(inout Ray original, in int idx) {
    _collect(original);

    bool invalidRay = 
        original.actived < 1 || 
        original.bounce <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    if (invalidRay) {
    //if (allInvocationsARB(invalidRay)) {
        return -1; 
    }

    int rayIndex = -1;
    int iterations = 1;
    int freed = 0;
    
    while (freed >= 0 && iterations >= 0) {
        iterations--;

        atomicMax(arcounter.Ut, 0); // prevent most decreasing
        int freed = atomicAdd(arcounter.Ut, -1)-1;
        atomicMax(arcounter.Ut, 0); // prevent most decreasing

        if (
            freed >= 0 && 
            availBuf.indc[freed] != 0xFFFFFFFF && 
            availBuf.indc[freed] != 0 && 
            availBuf.indc[freed] != -1
        ) {
            rayIndex = availBuf.indc[freed];
            break;
        }
    }

    //if (anyInvocationARB(rayIndex == -1)) {
    if (rayIndex == -1) {
        rayIndex = atomicAdd(arcounter.Rt, 1);
    }

    return createRayStrict(original, idx, rayIndex);
}

int createRayIdx(inout Ray original, in int idx, in int rayIndex) {
    _collect(original);

    bool invalidRay = 
        original.actived < 1 || 
        original.bounce <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    //if (allInvocationsARB(invalidRay)) {
    if (invalidRay) {
        return -1; 
    }
    
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


#endif
