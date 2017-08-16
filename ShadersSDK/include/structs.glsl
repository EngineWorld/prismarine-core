#ifndef _STRUCTS_H
#define _STRUCTS_H

#include "../include/mathlib.glsl"

struct Texel {
    vec4 coord;
    ivec4 EXT;
};

struct bbox {
#ifdef USE_WARP_OPTIMIZED
    float mn[4];
    float mx[4];
#else
    vec4 mn;
    vec4 mx;
#endif
};


// ray bitfield spec
// {0     }[1] - actived or not
// {1 ..2 }[2] - ray type (for example diffuse, specular, shadow)
// {3     }[1] - applyable direct light (can intersect with light data or not)
// {4 ..7 }[4] - target light index (for shadow type)
// {8 ..11}[4] - bounce index

struct RayRework {
    vec4 origin;
    vec4 direct;
    vec4 color;
    vec4 final;
    int bitfield; // up to 32-bits
    int idx; // ray index itself
    int texel; // texel index
    int hit; // index of hit chain
};

struct HitRework {
    vec4 uvt; // UV, distance, triangle
    vec4 albedo;
    vec4 metallicRoughness; // Y - roughtness, Z - metallic, also available other params
    vec4 normalHeight; // normal with height mapping, will already interpolated with geometry
    vec4 emission;
    vec4 texcoord;
    vec4 tangent;
    int bitfield; 
    int ray; // ray index
    int materialID;
    int next;
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
    bbox box;
#ifdef USE_WARP_OPTIMIZED
    int pdata[4];
#else
    ivec4 pdata;
#endif
};

struct VboDataStride {
    vec4 vertex;
    vec4 normal;
    vec4 texcoord;
    vec4 color;
    vec4 modifiers;
};

struct ColorChain {
    vec4 color;
    ivec4 cdata;
};



struct GroupFoundResult {
    int nextResult;
    float boxDistance;
    ivec2 range;
};

struct MeshUniformStruct {
    int vertexAccessor;
    int normalAccessor;
    int texcoordAccessor;
    int modifierAccessor;

    mat4 transform;
    mat4 transformInv;

    int materialID;
    int isIndexed;
    int nodeCount;
    int primitiveType;

    int loadingOffset;
    int storingOffset;
    int _reserved0;
    int _reserved1;
};

struct VirtualAccessor {
    int offset;
    int stride;
    int components;
    int type; // 0 is float, 1 is uint, 2 is 16bit uint
};



#endif
