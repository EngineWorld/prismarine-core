
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


#ifndef _RAYS_H
#define _RAYS_H

#include "../include/mathlib.glsl"
#include "../include/morton.glsl"
#include "../include/ballotlib.glsl"

layout ( std430, binding = 0 ) restrict buffer RaysSSBO { RayRework nodes[]; } rayBuf;
layout ( std430, binding = 1 ) restrict buffer HitsSSBO { HitRework nodes[]; } hitBuf;

#ifndef SIMPLIFIED_RAY_MANAGMENT
layout ( std430, binding = 2 ) restrict buffer TexelsSSBO { Texel nodes[]; } texelBuf;
layout ( std430, binding = 3 ) restrict buffer ColorChainBlock { ColorChain chains[]; } chBuf;
#endif

// current list
layout ( std430, binding = 4 ) readonly buffer ActivedIndicesSSBO { int indc[]; } activedBuf;

#ifndef SIMPLIFIED_RAY_MANAGMENT
layout ( std430, binding = 5 ) readonly buffer AvailablesIndicesSSBO { int indc[]; } availBuf;
#endif

// new list
#ifndef SIMPLIFIED_RAY_MANAGMENT
layout ( std430, binding = 6 ) restrict buffer CollectedActivesSSBO { int indc[]; } collBuf;
layout ( std430, binding = 7 ) restrict buffer FreedomIndicesSSBO { int indc[]; } freedBuf;
#endif

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

void _collect(inout RayRework ray){
#ifndef SIMPLIFIED_RAY_MANAGMENT
    vec4 color = min(max(ray.final, vec4(0.f)), vec4(1000.f));
    int idx = atomicIncCt(true); // allocate new index
    atomicCompSwap(texelBuf.nodes[ray.texel].EXT.y, -1, idx); // link first index

    // create new chain
    ColorChain cchain = chBuf.chains[idx];
    cchain.cdata.x = -1;
    cchain.color = vec4(color.xyz, 1.0f);
    chBuf.chains[idx] = cchain;
    ray.final.xyzw = vec4(0.0f);

    // link with previous (need do after)
    int prev = atomicExchange(texelBuf.nodes[ray.texel].EXT.z, idx);
    if (prev != -1) atomicExchange(chBuf.chains[prev].cdata.x, idx);
#endif
}

int addRayToList(in RayRework ray){
    int rayIndex = ray.idx;
    int actived = -1;

#ifndef SIMPLIFIED_RAY_MANAGMENT
    // ordered form list
    if (RayActived(ray) == 1) {
        int act = atomicIncAt(true);
        collBuf.indc[act] = rayIndex; actived = act;
    } else { // if not actived, why need?
        int freed = atomicIncQt(true);
        freedBuf.indc[freed] = rayIndex;
    }
#endif

    return actived;
}

int addRayToList(in RayRework ray, in int act){
    int rayIndex = ray.idx;
    int actived = -1;
#ifndef SIMPLIFIED_RAY_MANAGMENT
    if (RayActived(ray) == 1) {
        collBuf.indc[act] = rayIndex; actived = act;
    }
#endif
    return actived;
}

void storeRay(in int rayIndex, inout RayRework ray) {
    if (rayIndex == -1 || rayIndex == LONGEST || rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit) {
        RayActived(ray, 0);
    } else {
        if (RayActived(ray) == 0) {
            _collect(ray);
        }
        ray.idx = rayIndex;
        rayBuf.nodes[rayIndex] = ray;
    }
}

void storeRay(inout RayRework ray) {
    storeRay(ray.idx, ray);
}



#ifndef SIMPLIFIED_RAY_MANAGMENT
int createRayStrict(inout RayRework original, in int idx, in int rayIndex) {
    bool invalidRay = 
        (rayIndex == -1 || 
         rayIndex == LONGEST || 
         rayIndex >= RAY_BLOCK samplerUniform.currentRayLimit || 

        RayActived(original) == 0 || 
        RayBounce(original) <= 0 || 
        mlength(original.color.xyz) < 0.0001f);

    if (!invalidRay) {
        RayRework ray = original;
        int bounce = RayBounce(ray)-1;
        if (bounce < 0) {
            RayActived(ray, 0); 
        } else {
            RayBounce(ray, bounce >= 0 ? bounce : 0);
            ray.idx = rayIndex;
            ray.texel = idx;
            ray.hit = -1;
            rayBuf.nodes[rayIndex] = ray;
            addRayToList(ray);
        }
    }

    return rayIndex;
}

int createRayStrict(inout RayRework original, in int rayIndex) {
    return createRayStrict(original, original.texel, rayIndex);
}

int createRay(inout RayRework original, in int idx) {
    bool invalidRay = 
        RayActived(original) == 0 || 
        RayBounce(original) <= 0 || 
        mlength(original.color.xyz) < 0.0001f;

    if (mlength(original.final.xyz) >= 0.0001f && RayActived(original) == 0) {
        _collect(original);
    }

    int rayIndex = -1;
    if (!invalidRay) {
        int iterations = 0;
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

    RayBasis(original, 0); // is not basis ray
    return createRayStrict(original, idx, rayIndex);
}

int createRayIdx(inout RayRework original, in int idx, in int rayIndex) {
    bool invalidRay = 
        (RayActived(original) == 0 || 
         RayBounce(original) <= 0 || 
         mlength(original.color.xyz) < 0.0001f);

    if (invalidRay) {
        rayIndex = -1;
    } else {
        atomicMax(arcounter.Rt, rayIndex+1);

        if (mlength(original.final.xyz) >= 0.0001f && RayActived(original) == 0) {
            _collect(original);
        }
    }

    return createRayStrict(original, idx, rayIndex);
}

int createRay(in RayRework original) {
    return createRay(original, original.texel);
}

int createRay(in int idx) {
    RayRework ray;
    return createRay(ray, idx);
}

#endif



#endif
