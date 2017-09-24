
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


#ifndef _STRUCTS_H
#define _STRUCTS_H

#include "../include/mathlib.glsl"

struct _ext4 {
    int x;
    int y;
    int z;
    int w;
};

struct Texel {
     vec4 coord;
     vec4 last3d;
     _ext4 EXT;
};

struct bbox {
     vec4 mn;
     vec4 mx;
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
     vec4 normalHeight; // normal with height mapping, will already interpolated with geometry
     vec4 tangent; // also have 4th extra slot
     vec4 texcoord; // critical texcoords 

#ifdef ENABLE_AMD_INSTRUCTION_SET
    //f16vec4 metallicRoughness;
    //f16vec4 unk16;
    // f16vec4 emission;
    // f16vec4 albedo;
    uint64_t metallicRoughness;
    uint64_t unk16;
    uint64_t emission;
    uint64_t albedo;
#else
    uvec4 metallicRoughness; // 8 of 16-bit float, you can pack non-critical surface data
     uvec2 emission;
     uvec2 albedo;
#endif

    // integer metadata
     int bitfield; 
     int ray; // ray index
     int materialID;
     int next;
};



const ivec2 ACTIVED = ivec2(0, 1);
const ivec2 TYPE = ivec2(1, 2);
const ivec2 DIRECT_LIGHT = ivec2(3, 1);
const ivec2 TARGET_LIGHT = ivec2(4, 4);
const ivec2 BOUNCE = ivec2(8, 4);
const ivec2 BASIS = ivec2(12, 1);

int parameteri(const ivec2 parameter, inout int bitfield){
    return BFE(bitfield, parameter.x, parameter.y);
}

int parameteri(const ivec2 parameter, inout HitRework hit){
    return BFE(hit.bitfield, parameter.x, parameter.y);
}

int parameteri(const ivec2 parameter, inout RayRework ray){
    return BFE(ray.bitfield, parameter.x, parameter.y);
}

void parameteri(const ivec2 parameter, inout int bitfield, in int value){
    bitfield = BFI_HW(bitfield, value, parameter.x, parameter.y);
}

void parameteri(const ivec2 parameter, inout RayRework ray, in int value){
    ray.bitfield = BFI_HW(ray.bitfield, value, parameter.x, parameter.y);
}

void parameteri(const ivec2 parameter, inout HitRework hit, in int value){
    hit.bitfield = BFI_HW(hit.bitfield, value, parameter.x, parameter.y);
}




int HitActived(inout HitRework hit){
    return parameteri(ACTIVED, hit);
}

void HitActived(inout HitRework hit, in int actived){
    parameteri(ACTIVED, hit, actived);
}

int RayActived(inout RayRework ray){
    return parameteri(ACTIVED, ray);
}

void RayActived(inout RayRework ray, in int actived){
    parameteri(ACTIVED, ray, actived);
}

int RayType(inout RayRework ray){
    return parameteri(TYPE, ray);
}

void RayType(inout RayRework ray, in int type){
    parameteri(TYPE, ray, type);
}

int RayDL(inout RayRework ray){
    return parameteri(DIRECT_LIGHT, ray);
}

void RayDL(inout RayRework ray, in int dl){
    parameteri(DIRECT_LIGHT, ray, dl);
}

int RayTargetLight(inout RayRework ray){
    return parameteri(TARGET_LIGHT, ray);
}

void RayTargetLight(inout RayRework ray, in int tl){
    parameteri(TARGET_LIGHT, ray, tl);
}

int RayBounce(inout RayRework ray){
    return parameteri(BOUNCE, ray);
}

void RayBounce(inout RayRework ray, in int bn){
    parameteri(BOUNCE, ray, bn);
}

int RayBasis(inout RayRework ray){
    return parameteri(BASIS, ray);
}

void RayBasis(inout RayRework ray, in int basis){
    parameteri(BASIS, ray, basis);
}



struct HlbvhNode {
     bbox box;
#ifdef _ORDERED_ACCESS
     int branch[2];
     ivec2 pdata;
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

struct VirtualBufferView {
     int offset4;
     int stride4;
};

struct VirtualAccessor {
     int offset4;
     int bitfield;
     int bufferView;
};


int parameteri(const ivec2 parameter, inout VirtualAccessor vac){
    return BFE(vac.bitfield, parameter.x, parameter.y);
}

const ivec2 COMPONENTS = ivec2(0, 2);
const ivec2 ATYPE = ivec2(2, 4);
const ivec2 NORMALIZED = ivec2(6, 1);

int aComponents(inout VirtualAccessor vac) {
    return parameteri(COMPONENTS, vac);
}

int aType(inout VirtualAccessor vac) {
    return parameteri(ATYPE, vac);
}

int aNormalized(inout VirtualAccessor vac) {
    return parameteri(NORMALIZED, vac);
}



#endif
