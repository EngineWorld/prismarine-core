#ifndef _STRUCTS_H
#define _STRUCTS_H

#include "../include/mathlib.glsl"

struct Texel {
    highp vec4 coord;
    highp ivec4 EXT;
};

struct bbox {
#ifdef USE_WARP_OPTIMIZED
    highp float mn[4];
    highp float mx[4];
#else
    highp vec4 mn;
    highp vec4 mx;
#endif
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
    highp vec4 emission;
    highp vec4 albedo;

    // integer metadata
    highp int bitfield; 
    highp int ray; // ray index
    highp int materialID;
    highp int next;
};




int RayActived(in RayRework ray){
    return BFE(ray.bitfield, 0, 1);
}

void RayActived(inout RayRework ray, in int actived){
    ray.bitfield = BFI(ray.bitfield, actived, 0, 1);
}


int RayType(in RayRework ray){
    return BFE(ray.bitfield, 1, 2);
}

void RayType(inout RayRework ray, in int type){
    ray.bitfield = BFI(ray.bitfield, type, 1, 2);
}


int RayDL(in RayRework ray){
    return BFE(ray.bitfield, 3, 1);
}

void RayDL(inout RayRework ray, in int dl){
    ray.bitfield = BFI(ray.bitfield, dl, 3, 1);
}


int RayTargetLight(in RayRework ray){
    return BFE(ray.bitfield, 4, 4);
}

void RayTargetLight(inout RayRework ray, in int tl){
    ray.bitfield = BFI(ray.bitfield, tl, 4, 4);
}


int RayBounce(in RayRework ray){
    return BFE(ray.bitfield, 8, 4);
}

void RayBounce(inout RayRework ray, in int bn){
    ray.bitfield = BFI(ray.bitfield, bn, 8, 4);
}




struct HlbvhNode {
    highp bbox box;
#ifdef USE_WARP_OPTIMIZED
    highp int pdata[4];
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

struct VirtualAccessor {
    highp int offset;
    highp int stride;
    highp int components;
    highp int type; // 0 is float, 1 is uint, 2 is 16bit uint
};



#endif
