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


// unorm compasion
bool iseq8(float a, float cmp){ return abs(fma(a, 255.f, -cmp)) < 0.00001f; }
bool iseq16(float a, float cmp){ return abs(fma(a, 65535.f, -cmp)) < 0.00001f; }


// memory managment
void swap(inout int a, inout int b){ int t = a; a = b; b = t; }
uint exchange(inout uint mem, in uint v){ uint tmp = mem; mem = v; return tmp; }
int exchange(inout int mem, in int v){ int tmp = mem; mem = v; return tmp; }


// logical functions
bvec2 not2(in bvec2 a) { return bvec2(!a.x, !a.y); }
bvec2 and2(in bvec2 a, in bvec2 b) { return bvec2(a.x && b.x, a.y && b.y); }
bvec2 or2(in bvec2 a, in bvec2 b) { return bvec2(a.x || b.x, a.y || b.y); }


// mixing functions
void mixed(inout vec3 src, inout vec3 dst, in float coef){ dst *= coef; src *= 1.0f - coef; }
void mixed(inout vec3 src, inout vec3 dst, in vec3 coef){ dst *= coef; src *= 1.0f - coef; }


// matrix math
vec4 mult4(in vec4 vec, in mat4 mat){
    return vec4(dot(mat[0], vec), dot(mat[1], vec), dot(mat[2], vec), dot(mat[3], vec));
}

vec4 mult4(in mat4 tmat, in vec4 vec){
    return fma(tmat[0], vec.xxxx, fma(tmat[1], vec.yyyy, fma(tmat[2], vec.zzzz, tmat[3] * vec.wwww)));
}


int modi(in int a, in int b){
    return (a % b + b) % b;
}



// ordered increment
#if (!defined(FRAGMENT_SHADER) && !defined(ORDERING_NOT_REQUIRED))

// ballot math (alpha version)
#define WARP_SIZE 32
#ifdef EMULATE_BALLOT

#define   LC_IDX (gl_LocalInvocationID.x / WARP_SIZE)
#define LANE_IDX (gl_LocalInvocationID.x % WARP_SIZE)
#define UVEC_BALLOT_WARP uint

shared uint ballotCache[WORK_SIZE];
shared uint invocationCache[WORK_SIZE][WARP_SIZE];

uint genLtMask(){
    return (1 << LANE_IDX)-1;
}

uint ballot(in bool val) {
    ballotCache[LC_IDX] = 0;
    // warp can be have barrier, but is not required
    atomicOr(ballotCache[LC_IDX], uint(val) << LANE_IDX);
    // warp can be have barrier, but is not required
    return ballotCache[LC_IDX];
}

uint bitCount64(in uint a64) {
    return bitCount(a64);
}

uint readLane(in uint val, in int lane) {
    // warp can be have barrier, but is not required
    atomicExchange(invocationCache[LC_IDX][LANE_IDX], val);
    // warp can be have barrier, but is not required
    return invocationCache[LC_IDX][lane];
}

int readLane(in int val, in int lane) {
    // warp can be have barrier, but is not required
    atomicExchange(invocationCache[LC_IDX][LANE_IDX], uint(val));
    // warp can be have barrier, but is not required
    return int(invocationCache[LC_IDX][lane]);
}

int firstActive(){
    UVEC_BALLOT_WARP bits = ballot(true); return modi(findMSB(bits.x), WARP_SIZE);
}

#else

#define   LC_IDX (gl_LocalInvocationID.x / gl_SubGroupSizeARB)
#define LANE_IDX (gl_SubGroupInvocationARB)
#define UVEC_BALLOT_WARP uvec2

uvec2 genLtMask(){
    return unpackUint2x32(gl_SubGroupLtMaskARB);
}

uint bitCount64(in uvec2 lh) {
    //return bitCount(lh.x) + bitCount(lh.y);
    return bitCount(lh.x);
}

uint readLane(in uint val, in int lane){
    return readInvocationARB(val, lane);
}

int readLane(in int val, in int lane){
    return readInvocationARB(val, lane);
}

uvec2 ballot(in bool val) {
    return unpackUint2x32(ballotARB(val));
}

int firstActive(){
    UVEC_BALLOT_WARP bits = ballot(true);
    int msb = findMSB(bits.x);
    return modi(msb >= 0 ? msb : findMSB(bits.y), int(gl_SubGroupSizeARB));
}

#endif

uint makeOrder(in bool value){
    UVEC_BALLOT_WARP bits = ballot(value);
    return bitCount64(genLtMask() & bits);
}

uint countValid(in bool value){
    UVEC_BALLOT_WARP bits = ballot(value);
    return bitCount64(bits);
}

#define atomicIncWarpOrdered(mem, value, T) (readLane(atomicAdd(mem, mix(0, T(countValid(value)), LANE_IDX == firstActive())), firstActive()) + T(makeOrder(value)))

#else

#define atomicIncWarpOrdered(mem, value, T) atomicAdd(mem, T(value))

#endif

#endif