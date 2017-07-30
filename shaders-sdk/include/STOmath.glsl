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
    return fma(tmat[0], vec.xxxx, fma(tmat[1], vec.yyyy, fma(tmat[2], vec.zzzz, tmat[3] * vec.wwww)));
}

void swap(inout int a, inout int b){
    int t = a; a = b; b = t;
}

uint exchange(inout uint mem, in uint v){
     uint tmp = mem; mem = v; return tmp;
}

int exchange(inout int mem, in int v){
     int tmp = mem; mem = v; return tmp;
}

#endif