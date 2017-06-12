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



#endif
