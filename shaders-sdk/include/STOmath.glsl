#ifndef _STOMATH_H
#define _STOMATH_H


bool lessEqualF(in float a, in float b){
    return (b-a) > -PZERO;
}

bool lessF(in float a, in float b){
    return (b-a) >= PZERO;
}

bool greaterEqualF(in float a, in float b){
    return (a-b) > -PZERO;
}

bool greaterF(in float a, in float b){
    return (a-b) >= PZERO;
}

bool equalF(in float a, in float b){
    return abs(a-b) < PZERO;
}

float sqlen(in vec3 a) {
    return dot(a, a);
}

float sqlen(in vec2 a) {
    return dot(a, a);
}

float sqlen(in float v) {
    return v * v;
}

float mlength(in vec3 mcolor){
    return max(mcolor.x, max(mcolor.y, mcolor.z));
}

bool iseq8(float a, float cmp){
    return abs(fma(a, 255.f, -cmp)) < 0.00001f;
}

bool iseq16(float a, float cmp){
    return abs(fma(a, 65535.f, -cmp)) < 0.00001f;
}

vec4 divW(in vec4 aw){
    return aw / aw.w;
}


// hardware matrix multiply transposed (experimental)
vec4 mult4(in vec4 vec, in mat4 mat){
    return vec4(
        dot(mat[0], vec),
        dot(mat[1], vec),
        dot(mat[2], vec),
        dot(mat[3], vec)
    );
}

// hardware matrix multiply (experimental)
vec4 mult4(in mat4 tmat, in vec4 vec){
    // FMA stride version
    return fma(tmat[0], vec.xxxx, fma(tmat[1], vec.yyyy, fma(tmat[2], vec.zzzz, tmat[3] * vec.wwww)));
}



void swap(inout int a, inout int b){
    const int t = a;
    a = b;
    b = t;
}

uint exchange(inout uint mem, in uint v){
    const uint tmp = mem; mem = v; return tmp;
}

int exchange(inout int mem, in int v){
    const int tmp = mem; mem = v; return tmp;
}





#define ovec3 vec3 // use same register space

/*
struct ovec3 {
    float x;
    float y;
    float z;
};
*/

float length3v(in ovec3 a){
    return sqrt(fma(a.x, a.x, fma(a.y, a.y, a.z * a.z)));
}

float dot3v(in ovec3 a, in ovec3 b){
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}


ovec3 max3v(in ovec3 a, in ovec3 b){
    return ovec3(
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z)
    );
}

ovec3 min3v(in ovec3 a, in ovec3 b){
    return ovec3(
        min(a.x, b.x),
        min(a.y, b.y),
        min(a.z, b.z)
    );
}

ovec3 mul(in ovec3 a, in ovec3 b){
    return ovec3(
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    );
}

ovec3 add(in ovec3 a, in ovec3 b){
    return ovec3(
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

ovec3 neg(in ovec3 a){
    return ovec3(-a.x, -a.y, -a.z);
}

ovec3 div(in ovec3 a, in ovec3 b){
    return ovec3(
        a.x / b.x,
        a.y / b.y,
        a.z / b.z
    );
}

ovec3 divs(in ovec3 a, in float b){
    return ovec3(
        a.x / b,
        a.y / b,
        a.z / b
    );
}

ovec3 sdiv(in float a, in ovec3 b){
    return ovec3(
        a / b.x,
        a / b.y,
        a / b.z
    );
}

ovec3 sub(in ovec3 a, in ovec3 b){
    return ovec3(
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    );
}

ovec3 cross3v(in ovec3 a, in ovec3 b){
    return ovec3(
        fma(a.y, b.z, -(a.z * b.y)),
        fma(a.z, b.x, -(a.x * b.z)),
        fma(a.x, b.y, -(a.y * b.x))
    );
}

ovec3 fromVec3(in vec3 a){
    return a;
    //return ovec3(a.x, a.y, a.z);
}









// for unsupported systems


// 4x8

#ifdef ENABLE_UNSUPPOTED_FUNCTIONS

int packInt4x8(in ivec4 u8){
    int base = 0;
    base = bitfieldInsert(base, u8.x, 0, 8);
    base = bitfieldInsert(base, u8.y, 8, 8);
    base = bitfieldInsert(base, u8.z, 16, 8);
    base = bitfieldInsert(base, u8.w, 24, 8);
    return base;
}

uint packUint4x8(in uvec4 u8){
    uint base = 0;
    base = bitfieldInsert(base, u8.x, 0, 8);
    base = bitfieldInsert(base, u8.y, 8, 8);
    base = bitfieldInsert(base, u8.z, 16, 8);
    base = bitfieldInsert(base, u8.w, 24, 8);
    return base;
}


ivec4 unpackInt4x8(in int base){
    return ivec4(
        bitfieldExtract(base, 0, 8),
        bitfieldExtract(base, 8, 8),
        bitfieldExtract(base, 16, 8),
        bitfieldExtract(base, 24, 8)
    );
}

uvec4 unpackUint4x8(in uint base){
    return uvec4(
        bitfieldExtract(base, 0, 8),
        bitfieldExtract(base, 8, 8),
        bitfieldExtract(base, 16, 8),
        bitfieldExtract(base, 24, 8)
    );
}

// 2x16

#ifndef ENABLE_AMD_INSTRUCTION_SET

int packInt2x16(in ivec2 u16){
    int base = 0;
    base = bitfieldInsert(base, u16.x, 0, 16);
    base = bitfieldInsert(base, u16.y, 16, 16);
    return base;
}

uint packUint2x16(in uvec2 u16){
    uint base = 0;
    base = bitfieldInsert(base, u16.x, 0, 16);
    base = bitfieldInsert(base, u16.y, 16, 16);
    return base;
}

ivec2 unpackInt2x16(in int base){
    return ivec2(
        bitfieldExtract(base, 0, 16),
        bitfieldExtract(base, 16, 16)
    );
}

uvec2 unpackUint2x16(in uint base){
    return uvec2(
        bitfieldExtract(base, 0, 16),
        bitfieldExtract(base, 16, 16)
    );
}

#endif
#endif

#endif
