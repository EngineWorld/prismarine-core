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
