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



// other parameters TODO
struct SurfaceData {
    vec3 normal;
    float height;

    vec4 emission;
    vec4 albedo;
    vec4 specular;
    
    float metallic;
    float roughness;

    int culling;
};



#endif
