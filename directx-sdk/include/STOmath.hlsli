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

float sqlen(in float3 a) {
    return dot(a, a);
}

float sqlen(in float2 a) {
    return dot(a, a);
}

float sqlen(in float v) {
    return v * v;
}

float mlength(in float3 mcolor){
    return max(mcolor.x, max(mcolor.y, mcolor.z));
}

float4 divW(in float4 aw){
    return aw / aw.w;
}


void swap(inout int a, inout int b){
    int t = a;
    a = b;
    b = t;
}

uint exchange(inout uint mem, in uint v){
     uint tmp = mem; mem = v; return tmp;
}

int exchange(inout int mem, in int v){
     int tmp = mem; mem = v; return tmp;
}

#endif
