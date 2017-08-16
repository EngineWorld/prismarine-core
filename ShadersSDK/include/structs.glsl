#ifndef _STRUCTS_H
#define _STRUCTS_H

#define Vc1 float
#define Vc2 vec2
#define Vc3 Stride3f
#define Vc4 vec4
#define Vc4x4 mat4
#define iVc1 int
#define iVc2 ivec2
#define iVc3 Stride3i
#define iVc4 ivec4

struct Stride4f {
    float x;
    float y;
    float z;
    float w;
};

struct Stride4i {
    int x;
    int y;
    int z;
    int w;
};

struct Stride2f {
    float x;
    float y;
};

struct Stride3f {
    float x;
    float y;
    float z;
};

struct Stride3i {
    int x;
    int y;
    int z;
};

vec2 toVec2(in Stride2f a){
    return vec2(a.x, a.y);
}

vec3 toVec3(in Stride3f a){
    return vec3(a.x, a.y, a.z);
}

vec4 toVec4(in Stride4f a){
    return vec4(a.x, a.y, a.z, a.w);
}

ivec3 toVec3(in Stride3i a){
    return ivec3(a.x, a.y, a.z);
}

Stride2f toStride2(in vec2 a){
    Stride2f o;
    o.x = a.x;
    o.y = a.y;
    return o;
}

Stride3f toStride3(in vec3 a){
    Stride3f o;
    o.x = a.x;
    o.y = a.y;
    o.z = a.z;
    return o;
}

Stride4f toStride4(in vec4 a){
    Stride4f o;
    o.x = a.x;
    o.y = a.y;
    o.z = a.z;
    o.w = a.w;
    return o;
}

Stride4i toStride4(in ivec4 a){
    Stride4i o;
    o.x = a.x;
    o.y = a.y;
    o.z = a.z;
    o.w = a.w;
    return o;
}

struct Texel {
    Vc4 coord;
    iVc4 EXT;
};

struct bbox {
#ifdef USE_WARP_OPTIMIZED
    Vc1 mn[4];
    Vc1 mx[4];
#else
    Vc4 mn;
    Vc4 mx;
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


#define BFE(a,o,n) ((a >> o) & ((1 << n)-1))

int BFI(in int base, in int inserts, in int offset, in int bits){
    int mask = bits >= 32 ? 0xFFFFFFFF : (1<<bits)-1;
    int offsetMask = mask << offset;
    return ((base & (~offsetMask)) | ((inserts & mask) << offset));
}


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






struct Ray {
#ifdef USE_WARP_OPTIMIZED
    Vc1 origin[4];
    Vc1 direct[4];
#else
    Vc4 origin;
    Vc4 direct;
#endif
    Vc4 color;
    Vc4 final;
    iVc4 params;
    iVc1 idx;
    iVc1 bounce;
    iVc1 texel;
    iVc1 actived;
    // planned additional block for hit chains
};

struct Hit {
    Vc4 normal;
    Vc4 tangent;
    Vc4 texcoord;
    Vc4 vcolor;
    Vc4 vmods;
    Vc1 dist;
    iVc1 triangleID;
    iVc1 materialID;
    iVc1 shaded;
};

struct HlbvhNode {
    bbox box;
#ifdef USE_WARP_OPTIMIZED
    iVc1 pdata[4];
#else
    iVc4 pdata;
#endif
    //iVc4 leading;
};

struct VboDataStride {
    Vc4 vertex;
    Vc4 normal;
    Vc4 texcoord;
    Vc4 color;
    Vc4 modifiers;
};

struct ColorChain {
    Vc4 color;
    iVc4 cdata;
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
