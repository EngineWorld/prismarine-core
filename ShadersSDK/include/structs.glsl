
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


#ifndef _STRUCTS_H
#define _STRUCTS_H

#include "../include/mathlib.glsl"

struct Texel {
    highp vec4 coord;
    highp vec4 last3d;
    highp ivec4 EXT;
};

struct bbox {
    highp vec4 mn;
    highp vec4 mx;
};


// ray bitfield spec
// {0     }[1] - actived or not
// {1 ..2 }[2] - ray type (for example diffuse, specular, shadow)
// {3     }[1] - applyable direct light (can intersect with light data or not)
// {4 ..7 }[4] - target light index (for shadow type)
// {8 ..11}[4] - bounce index

struct RayRework {
    highp vec4 origin;
    highp vec4 direct;
    highp vec4 color;
    highp vec4 final;
    highp int bitfield; // up to 32-bits
    highp int idx; // ray index itself
    highp int texel; // texel index
    highp int hit; // index of hit chain
};

struct HitRework {
    highp vec4 uvt; // UV, distance, triangle
    highp vec4 normalHeight; // normal with height mapping, will already interpolated with geometry
    highp vec4 tangent; // also have 4th extra slot
    highp vec4 texcoord; // critical texcoords 

    // low four 16 bit - texcoords
    highp uvec4 metallicRoughness; // 8 of 16-bit float, you can pack non-critical surface data

    // color parameters
    //highp vec4 emission;
    //highp vec4 albedo;
    highp uvec2 emission;
    highp uvec2 albedo;

    // integer metadata
    highp int bitfield; 
    highp int ray; // ray index
    highp int materialID;
    highp int next;
};


int HitActived(inout HitRework hit){
    return BFE(hit.bitfield, 0, 1);
}

void HitActived(inout HitRework hit, in int actived){
    hit.bitfield = BFI_HW(hit.bitfield, actived, 0, 1);
}



int RayActived(inout RayRework ray){
    return BFE(ray.bitfield, 0, 1);
}

void RayActived(inout RayRework ray, in int actived){
    ray.bitfield = BFI_HW(ray.bitfield, actived, 0, 1);
}


int RayType(inout RayRework ray){
    return BFE(ray.bitfield, 1, 2);
}

void RayType(inout RayRework ray, in int type){
    ray.bitfield = BFI_HW(ray.bitfield, type, 1, 2);
}


int RayDL(inout RayRework ray){
    return BFE(ray.bitfield, 3, 1);
}

void RayDL(inout RayRework ray, in int dl){
    ray.bitfield = BFI_HW(ray.bitfield, dl, 3, 1);
}


int RayTargetLight(inout RayRework ray){
    return BFE(ray.bitfield, 4, 4);
}

void RayTargetLight(inout RayRework ray, in int tl){
    ray.bitfield = BFI_HW(ray.bitfield, tl, 4, 4);
}


int RayBounce(inout RayRework ray){
    return BFE(ray.bitfield, 8, 4);
}

void RayBounce(inout RayRework ray, in int bn){
    ray.bitfield = BFI_HW(ray.bitfield, bn, 8, 4);
}


int RayBasis(inout RayRework ray){
    return BFE(ray.bitfield, 12, 1);
}

void RayBasis(inout RayRework ray, in int basis){
    ray.bitfield = BFI_HW(ray.bitfield, basis, 12, 1);
}



struct HlbvhNode {
    highp bbox box;
#ifdef _ORDERED_ACCESS
    highp int branch[2];
    highp ivec2 pdata;
#else
    highp ivec4 pdata;
#endif
};

struct VboDataStride {
    highp vec4 vertex;
    highp vec4 normal;
    highp vec4 texcoord;
    highp vec4 color;
    highp vec4 modifiers;
};

struct ColorChain {
    highp vec4 color;
    highp ivec4 cdata;
};



struct GroupFoundResult {
    highp int nextResult;
    highp float boxDistance;
    highp ivec2 range;
};

struct MeshUniformStruct {
    highp int vertexAccessor;
    highp int normalAccessor;
    highp int texcoordAccessor;
    highp int modifierAccessor;

    highp mat4 transform;
    highp mat4 transformInv;

    highp int materialID;
    highp int isIndexed;
    highp int nodeCount;
    highp int primitiveType;

    highp int loadingOffset;
    highp int storingOffset;
    highp int _reserved0;
    highp int _reserved1;
};

struct VirtualBufferView {
    highp int offset4;
    highp int stride4;
};

struct VirtualAccessor {
    highp int offset4;
    highp int bitfield;
    highp int bufferView;
};

int aComponents(inout VirtualAccessor vac) {
    return BFE(vac.bitfield, 0, 2);
}

int aType(inout VirtualAccessor vac) {
    return BFE(vac.bitfield, 2, 4);
}

int aNormalized(inout VirtualAccessor vac) {
    return BFE(vac.bitfield, 6, 1);
}



#endif
