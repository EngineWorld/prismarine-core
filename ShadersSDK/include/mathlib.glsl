#ifndef _STOMATH_H
#define _STOMATH_H


// roundly comparsion functions
bool lessEqualF(in float a, in float b) { return (b-a) > -PZERO; }
bool lessF(in float a, in float b) { return (b-a) >= PZERO; }
bool greaterEqualF(in float a, in float b) { return (a-b) > -PZERO; }
bool greaterF(in float a, in float b) { return (a-b) >= PZERO; }
bool equalF(in float a, in float b) { return abs(a-b) < PZERO; }


// vector math utils
float sqlen(in vec3 a) { return dot(a, a); }
float sqlen(in vec2 a) { return dot(a, a); }
float sqlen(in float v) { return v * v; }
float mlength(in vec3 mcolor){ return max(mcolor.x, max(mcolor.y, mcolor.z)); }
vec4 divW(in vec4 aw){ return aw / aw.w; }


// memory managment
void swap(inout int a, inout int b){ int t = a; a = b; b = t; }
uint exchange(inout uint mem, in uint v){ uint tmp = mem; mem = v; return tmp; }
int exchange(inout int mem, in int v){ int tmp = mem; mem = v; return tmp; }
int add(inout int mem, in int v){ int tmp = mem; mem += v; return tmp; }

// logical functions
bvec2 not2(in bvec2 a) { return bvec2(!a.x, !a.y); }
bvec2 and2(in bvec2 a, in bvec2 b) { return bvec2(a.x && b.x, a.y && b.y); }
bvec2 or2(in bvec2 a, in bvec2 b) { return bvec2(a.x || b.x, a.y || b.y); }

// logical functions (bvec4)
bvec4 or(in bvec4 a, in bvec4 b){
    return bvec4(
        a.x || b.x,
        a.y || b.y,
        a.z || b.z,
        a.w || b.w
    );
}

bvec4 and(in bvec4 a, in bvec4 b){
    return bvec4(
        a.x && b.x,
        a.y && b.y,
        a.z && b.z,
        a.w && b.w
    );
}

bvec4 not(in bvec4 a){
    return bvec4(!a.x, !a.y, !a.z, !a.w);
}




// mixing functions
void mixed(inout vec3 src, inout vec3 dst, in float coef){ dst *= coef; src *= 1.0f - coef; }
void mixed(inout vec3 src, inout vec3 dst, in vec3 coef){ dst *= coef; src *= 1.0f - coef; }


// matrix math
vec4 mult4(in vec4 vec, in mat4 tmat){
    //return vec4(dot(tmat[0], vec), dot(tmat[1], vec), dot(tmat[2], vec), dot(tmat[3], vec));
    return vec * tmat;
}

vec4 mult4(in mat4 tmat, in vec4 vec){
    //return fma(tmat[0], vec.xxxx, fma(tmat[1], vec.yyyy, fma(tmat[2], vec.zzzz, tmat[3] * vec.wwww)));
    return tmat * vec;
}


int modi(in int a, in int b){
    return (a % b + b) % b;
}

uint btc(in uint vlc){
    return vlc == 0 ? 0 : bitCount(vlc);
}





float intersectCubeSingle(in vec3 origin, in vec3 ray, in vec4 cubeMin, in vec4 cubeMax, inout float near, inout float far) {
    vec3 dr = 1.0f / ray;
    vec3 tMin = (cubeMin.xyz - origin) * dr;
    vec3 tMax = (cubeMax.xyz - origin) * dr;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
#ifdef ENABLE_AMD_INSTRUCTION_SET
    float tNear = max3(t1.x, t1.y, t1.z);
    float tFar  = min3(t2.x, t2.y, t2.z);
#else
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);
#endif
    bool isCube = greaterEqualF(tFar, tNear) && greaterEqualF(tFar, 0.0f);
    float inf = INFINITY;
    near = isCube ? min(tNear, tFar) : inf;
    far  = isCube ? max(tNear, tFar) : inf;
    return (isCube ? (lessF(near, 0.0f) ? far : near) : inf);
}

void intersectCubeApart(in vec3 origin, in vec3 ray, in vec4 cubeMin, in vec4 cubeMax, inout float near, inout float far) {
    vec3 dr = 1.0f / ray;
    vec3 tMin = (cubeMin.xyz - origin) * dr;
    vec3 tMax = (cubeMax.xyz - origin) * dr;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
#ifdef ENABLE_AMD_INSTRUCTION_SET
    near = max3(t1.x, t1.y, t1.z);
    far  = min3(t2.x, t2.y, t2.z);
#else
    near = max(max(t1.x, t1.y), t1.z);
    far  = min(min(t2.x, t2.y), t2.z);
#endif
}





#define BFE(a,o,n) ((a >> o) & ((1 << n)-1))

uvec2 U2P(in uint64_t pckg) {
    return uvec2((pckg >> 0) & 0xFFFFFFFF, (pckg >> 32) & 0xFFFFFFFF);
}

int BFI(in int base, in int inserts, in int offset, in int bits){
    int mask = bits >= 32 ? 0xFFFFFFFF : (1<<bits)-1;
    int offsetMask = mask << offset;
    return ((base & (~offsetMask)) | ((inserts & mask) << offset));
}




// bit logic
const int bx = 1, by = 2, bz = 4, bw = 8;
const int bswiz[8] = {1, 2, 4, 8, 16, 32, 64, 128};

int cB(in bool a, in int swizzle){
    return a ? swizzle : 0;
}


int cB4(in bvec4 a){
    ivec4 mx4 = mix(ivec4(0), ivec4(bx, by, bz, bw), a);
    ivec2 mx2 = mx4.xy | mx4.wz; // swizzle or
    return (mx2.x | mx2.y); // rest of
}

int cB2(in bvec2 a){
    ivec2 mx = mix(ivec2(0), ivec2(bx, by), a);
    return (mx.x | mx.y);
}

bvec2 cI2(in int a){
    return bvec2(
        (a & bx) > 0, 
        (a & by) > 0
    );
}

bool anyB(in int a){
    return a > 0;
}

bool allB2(in int a){
    return a == 3;
}



#endif