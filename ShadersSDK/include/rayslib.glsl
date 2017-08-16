#ifndef _RAYS_H
#define _RAYS_H

#include "../include/mathlib.glsl"
#include "../include/morton.glsl"
#include "../include/ballotlib.glsl"

layout ( std430, binding = 0 ) restrict buffer RaysSSBO { RayRework nodes[]; } rayBuf;
layout ( std430, binding = 1 ) restrict buffer HitsSSBO { HitRework nodes[]; } hitBuf;
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

    // traverser counters
    int Ft;
    int Gt; 

    int Ht;
} arcounter;



initAtomicIncFunction(arcounter.At, atomicIncAt, int);
initAtomicIncFunction(arcounter.Rt, atomicIncRt, int);
initAtomicIncFunction(arcounter.Qt, atomicIncQt, int);
initAtomicDecFunction(arcounter.Ut, atomicDecUt, int);
initAtomicIncFunction(arcounter.Ct, atomicIncCt, int);
initAtomicIncFunction(arcounter.Ft, atomicIncFt, int);
initAtomicIncFunction(arcounter.Gt, atomicIncGt, int);
initAtomicIncFunction(arcounter.Ht, atomicIncHt, int);

void _collect(inout RayRework ray) {
    /*
    vec4 color = max(ray.final, vec4(0.f));
    int idx = atomicIncCt(true);
    int prev = atomicExchange(texelBuf.nodes[ray.texel].EXT.z, idx);
    bool isFirst = atomicCompSwap(texelBuf.nodes[ray.texel].EXT.y, -1, idx) == -1;
    if (prev != -1) atomicExchange(chBuf.chains[prev].cdata.x, idx); // linked 
    chBuf.chains[idx].color = vec4(color.xyz, 1.0f);
    atomicExchange(chBuf.chains[idx].cdata.x, -1);
    ray.final.xyzw = vec4(0.0f);
    */
    
    vec4 color = max(ray.final, vec4(0.f));
    float amplitude = mlength(color.xyz);
    int idx = atomicIncCt(true);
    int prev = atomicExchange(texelBuf.nodes[ray.texel].EXT.y, idx);
    ColorChain ch = chBuf.chains[idx];
    ch.color = vec4(color.xyz, 1.0f);
    ch.cdata = ivec4(prev, 0, 0, 0);
    chBuf.chains[idx] = ch;
    ray.final.xyzw = vec4(0.0f);
    
}

int addRayToList(in RayRework ray){
    int rayIndex = ray.idx;
    int actived = -1;

    // ordered form list
    if (RayActived(ray) == 1) {
        int act = atomicIncAt(true);
        collBuf.indc[act] = rayIndex; actived = act;
    } else { // if not actived, why need?
        int freed = atomicIncQt(true);
        freedBuf.indc[freed] = rayIndex;
    }

    return actived;
}

int addRayToList(in RayRework ray, in int act){
    int rayIndex = ray.idx;
    int actived = -1;
    if (RayActived(ray) == 1) {
        collBuf.indc[act] = rayIndex; actived = act;
    }
    return actived;
}

void storeRay(in int rayIndex, inout RayRework ray) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        RayActived(ray, 0);
    } else {
        _collect(ray);
        ray.idx = rayIndex;
        rayBuf.nodes[rayIndex] = ray;
    }
}

int createRayStrict(inout RayRework original, in int idx, in int rayIndex) {
    bool invalidRay = true && // debug
        (rayIndex == -1 || 
        rayIndex == LONGEST || 
        rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit || 

        RayActived(original) < 1 || 
        RayBounce(original) <= 0 || 
        mlength(original.color.xyz) < 0.0001f);

    if (!invalidRay) {
        RayRework ray = original;
        int bounce = RayBounce(ray)-1;
        RayBounce(ray, bounce > 0 ? bounce : 0);
        if (bounce < 0) RayActived(ray, 0); 
        ray.idx = rayIndex;
        ray.texel = idx;
        ray.hit = -1;
        rayBuf.nodes[rayIndex] = ray;
        addRayToList(ray);
    }

    return rayIndex;
}

int createRayStrict(inout RayRework original, in int rayIndex) {
    return createRayStrict(original, original.texel, rayIndex);
}

int createRay(inout RayRework original, in int idx) {
    bool invalidRay = 
        RayActived(original) < 1 || 
        RayBounce(original) <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    if (mlength(original.final.xyz) >= 0.0001f) {
        _collect(original);
    }

    int rayIndex = -1;
    if (!invalidRay) {
        int iterations = 1;
        int freed = 0;
        
        while (freed >= 0 && iterations >= 0) {
            iterations--;

            atomicMax(arcounter.Ut, 0); // prevent most decreasing
            int freed = max(atomicDecUt(true)-1, -1);
            atomicMax(arcounter.Ut, 0); // prevent most decreasing

            if (
                freed >= 0 && 
                availBuf.indc[freed] != 0 && 
                availBuf.indc[freed] != LONGEST && 
                availBuf.indc[freed] != -1
            ) {
                rayIndex = availBuf.indc[freed];
                break;
            }
        }

        if (rayIndex == -1) {
            rayIndex = atomicIncRt(true);
        }
    }
    return createRayStrict(original, idx, rayIndex);
}

int createRayIdx(inout RayRework original, in int idx, in int rayIndex) {
    bool invalidRay = true && 
        (RayActived(original) < 1 || 
        RayBounce(original) <= 0 || 
        mlength(original.color.xyz) < 0.0001f);

    if (mlength(original.final.xyz) >= 0.0001f) {
        _collect(original);
    }

    if (invalidRay) {
        rayIndex = -1;
    } else {
        atomicMax(arcounter.Rt, rayIndex+1);
    }

    return createRayStrict(original, idx, rayIndex);
}

void storeRay(inout RayRework ray) {
    storeRay(ray.idx, ray);
}

int createRay(in RayRework original) {
    return createRay(original, original.texel);
}

int createRay(in int idx) {
    RayRework ray;
    return createRay(ray, idx);
}

#endif
