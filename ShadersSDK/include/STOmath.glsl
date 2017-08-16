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


vec3 bboxNormal(in bbox box, in vec3 hit){
    vec3 c = (box.mn + box.mx).xyz * 0.5f;
    vec3 p = hit - c;
    vec3 d = abs((box.mx - box.mn).xyz * 0.5f);
    float bias = 1.000001f;
    return normalize(floor(p / d * bias));
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
shared uint invocationCache[WORK_SIZE/WARP_SIZE][WARP_SIZE];

uint genLtMask(){
    return (1 << LANE_IDX)-1;
}

uint ballotHW(in bool val) {
    ballotCache[LC_IDX] = 0;
    // warp can be have barrier, but is not required
    atomicOr(ballotCache[LC_IDX], uint(val) << LANE_IDX);
    // warp can be have barrier, but is not required
    return ballotCache[LC_IDX];
}

uint bitCount64(in uint a64) {
    return btc(a64);
}


float readLane(in float val, in int lane) {
    // warp can be have barrier, but is not required
    atomicExchange(invocationCache[LC_IDX][LANE_IDX], floatBitsToUint(val));
    // warp can be have barrier, but is not required
    return uintBitsToFloat(invocationCache[LC_IDX][lane]);
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
    return findLSB(ballotHW(true).x);
}

#else

#define   LC_IDX (gl_LocalInvocationID.x / gl_SubGroupSizeARB)
#define LANE_IDX (gl_LocalInvocationID.x % gl_SubGroupSizeARB)
#define UVEC_BALLOT_WARP uvec2

uvec2 genLtMask(){
    //return unpackUint2x32(gl_SubGroupLtMaskARB);

    uvec2 mask = uvec2(0, 0);
    if (LANE_IDX >= 64) {
        mask = uvec2(0xFFFFFFFF, 0xFFFFFFFF);
    } else 
    if (LANE_IDX >= 32) {
        mask = uvec2(0xFFFFFFFF, (1 << (LANE_IDX-32))-1);
    } else {
        mask = uvec2((1 << LANE_IDX)-1, 0);
    }
    return mask;
}

uint bitCount64(in uvec2 lh) {
    return btc(lh.x) + btc(lh.y);
}


vec4 readLane(in vec4 val, in int lane){
    return readInvocationARB(val, lane);
}

float readLane(in float val, in int lane){
    return readInvocationARB(val, lane);
}

uint readLane(in uint val, in int lane){
    return readInvocationARB(val, lane);
}

int readLane(in int val, in int lane){
    return readInvocationARB(val, lane);
}

uvec2 ballotHW(in bool val) {
    return unpackUint2x32(ballotARB(val)) & uvec2(
        gl_SubGroupSizeARB >= 32 ? 0xFFFFFFFF : ((1 << gl_SubGroupSizeARB)-1), 
        gl_SubGroupSizeARB >= 64 ? 0xFFFFFFFF : ((1 << (gl_SubGroupSizeARB-32))-1)
    );
}

int firstActive(){
    UVEC_BALLOT_WARP bits = ballotHW(true);
    int lv = findLSB(bits.x);
    return (lv >= 0 ? lv : (32+findLSB(bits.y)));
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

#define initAtomicIncFunction(mem, fname, T)\
T fname(in bool value){ \
    int activeLane = firstActive();\
    UVEC_BALLOT_WARP bits = ballotHW(value);\
    T sumInOrder = T(bitCount64(bits));\
    T idxInOrder = T(bitCount64(genLtMask() & bits));\
    return readLane(LANE_IDX == activeLane ? atomicAdd(mem,  mix(0, sumInOrder, LANE_IDX == activeLane)) : 0, activeLane) + idxInOrder; \
}

#define initAtomicDecFunction(mem, fname, T)\
T fname(in bool value){ \
    int activeLane = firstActive();\
    UVEC_BALLOT_WARP bits = ballotHW(value);\
    T sumInOrder = T(bitCount64(bits));\
    T idxInOrder = T(bitCount64(genLtMask() & bits));\
    return readLane(LANE_IDX == activeLane ? atomicAdd(mem, -mix(0, sumInOrder, LANE_IDX == activeLane)) : 0, activeLane) - idxInOrder; \
}

#else

#define initAtomicIncFunction(mem, fname, T)\
T fname(in bool value){ \
    return atomicAdd(mem, T(value)); \
}

#define initAtomicDecFunction(mem, fname, T)\
T fname(in bool value){ \
    return atomicAdd(mem, -T(value)); \
}

#endif


#define BFE(a,o,n) ((a >> o) & ((1 << n)-1))


#if defined(ENABLE_AMD_INSTRUCTION_SET) || defined(ENABLE_NVIDIA_INSTRUCTION_SET)
#define INDEX16 uint16_t
#define M16(m, i) (m[i])
#else
#define INDEX16 uint
#define M16(m, i) (BFE(m[i/2], int(16*(i%2)), 16))
#endif

#ifdef ENABLE_INT16_LOADING
#define INDICE_T INDEX16
#define PICK(m, i) M16(m, i)
#else
#define INDICE_T uint
#define PICK(m, i) m[i]
#endif


#endif