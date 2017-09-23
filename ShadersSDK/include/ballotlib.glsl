
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


// ordered increment
#if (!defined(FRAGMENT_SHADER) && !defined(ORDERING_NOT_REQUIRED))

// ballot math (alpha version)
#define WARP_SIZE 32
#ifdef EMULATE_BALLOT

#define   LC_IDX (gl_LocalInvocationID.x / WARP_SIZE)
#define LANE_IDX (gl_SubGroupInvocationARB)
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
    return U2P(gl_SubGroupLtMaskARB);
    /*
    uvec2 mask = uvec2(0, 0);
    if (gl_SubGroupInvocationARB >= 64u) {
        mask = uvec2(0xFFFFFFFFu, 0xFFFFFFFFu);
    } else 
    if (gl_SubGroupInvocationARB >= 32u && gl_SubGroupInvocationARB < 64u) {
        mask = uvec2(0xFFFFFFFFu, gl_SubGroupInvocationARB == 32 ? 0u : (1u << (gl_SubGroupInvocationARB-32u))-1u);
    } else 
    if (gl_SubGroupInvocationARB >= 0u && gl_SubGroupInvocationARB < 32u) {
        mask = uvec2(gl_SubGroupInvocationARB == 0 ? 0u : (1 << gl_SubGroupInvocationARB)-1, 0u);
    }
    return mask;*/
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
    return U2P(ballotARB(val));
}

int firstActive(){
    UVEC_BALLOT_WARP bits = ballotHW(true);
    int lv = lsb(bits.x);
    int hi = lsb(bits.y);
    return (lv >= 0) ? lv : (32 + hi);
}

#endif

#define initAtomicIncFunction(mem, fname, T)\
T fname(in bool value){ \
    int activeLane = firstActive();\
    UVEC_BALLOT_WARP bits = ballotHW(value);\
    T sumInOrder = T(bitCount64(bits));\
    T idxInOrder = T(bitCount64(genLtMask() & bits));\
    return readLane(LANE_IDX == activeLane ? (sumInOrder > 0 ? atomicAdd(mem,  mix(0, sumInOrder, LANE_IDX == activeLane)) : 0) : 0, activeLane) + idxInOrder; \
}

#define initAtomicIncFunctionMem(mem, fname, T)\
T fname(in bool value, in int memc){ \
    int activeLane = firstActive();\
    UVEC_BALLOT_WARP bits = ballotHW(value);\
    T sumInOrder = T(bitCount64(bits));\
    T idxInOrder = T(bitCount64(genLtMask() & bits));\
    return readLane(LANE_IDX == activeLane ? (sumInOrder > 0 ? atomicAdd(mem[memc], mix(0, sumInOrder, LANE_IDX == activeLane)) : 0) : 0, activeLane) + idxInOrder; \
}



#define initAtomicDecFunction(mem, fname, T)\
T fname(in bool value){ \
    int activeLane = firstActive();\
    UVEC_BALLOT_WARP bits = ballotHW(value);\
    T sumInOrder = T(bitCount64(bits));\
    T idxInOrder = T(bitCount64(genLtMask() & bits));\
    return readLane(LANE_IDX == activeLane ? (sumInOrder > 0 ? atomicAdd(mem, -mix(0, sumInOrder, LANE_IDX == activeLane)) : 0) : 0, activeLane) - idxInOrder; \
}

#define initAtomicDecFunctionMem(mem, fname, T)\
T fname(in bool value, in int memc){ \
    int activeLane = firstActive();\
    UVEC_BALLOT_WARP bits = ballotHW(value);\
    T sumInOrder = T(bitCount64(bits));\
    T idxInOrder = T(bitCount64(genLtMask() & bits));\
    return readLane(LANE_IDX == activeLane ? (sumInOrder > 0 ? atomicAdd(mem[memc], -mix(0, sumInOrder, LANE_IDX == activeLane)) : 0) : 0, activeLane) - idxInOrder; \
}

// with multiplier support
#define initAtomicIncByFunction(mem, fname, T)\
T fname(in bool value, const int by){ \
    int activeLane = firstActive();\
    UVEC_BALLOT_WARP bits = ballotHW(value);\
    T sumInOrder = T(bitCount64(bits));\
    T idxInOrder = T(bitCount64(genLtMask() & bits));\
    return readLane(LANE_IDX == activeLane ? (sumInOrder > 0 ? atomicAdd(mem, mix(0, sumInOrder * by, LANE_IDX == activeLane)) : 0) : 0, activeLane) + idxInOrder * by; \
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

