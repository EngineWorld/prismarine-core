
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md
//#define EMULATE_BALLOT

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_AMD_gpu_shader_int64 : enable

#ifndef EMULATE_BALLOT
#extension GL_ARB_shader_ballot : require
#endif

//#define BLOCK_SIZE 1024
#define BLOCK_SIZE 256
#define BLOCK_SIZE_RT (gl_WorkGroupSize.x)

#ifdef AMD_SUPPORT
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

#define WARP_SIZE_RT gl_SubGroupSizeARB

#define WORK_SIZE (BLOCK_SIZE/WARP_SIZE)
#define WORK_SIZE_RT (gl_WorkGroupSize.x / gl_SubGroupSizeARB)

//#define BITS_PER_PASS 4
//#define RADICES 16
//#define RADICES_MASK 0xf

#define BITS_PER_PASS 8
#define RADICES 256
#define RADICES_MASK 0xff

uint LC_IDX = 0;
uint LANE_IDX = 0;
uint LT_IDX = 0;

#define UVEC_WARP uint
#define BVEC_WARP bool
#define UVEC64_WARP uint64_t

//#define READ_LANE(V, I) ((I >= 0 && I < gl_SubGroupSizeARB) ? readLane(V, I) : 0)
#define READ_LANE(V, I) (uint(I >= 0 && I < gl_SubGroupSizeARB) * readLane(V, I))

#define BFE(a,o,n) ((a >> o) & ((1 << n)-1))

#define KEYTYPE UVEC64_WARP
//#define KEYTYPE UVEC_WARP
layout (std430, binding = 20) buffer KeyInBlock {KEYTYPE KeyIn[];};
layout (std430, binding = 21) buffer ValueInBlock {uint ValueIn[];};
layout (std430, binding = 24) buffer VarsBlock {
    uint NumKeys;
    uint Shift;
    uint Descending;
    uint IsSigned;
};
layout (std430, binding = 25) buffer KeyTmpBlock {KEYTYPE KeyTmp[];};
layout (std430, binding = 26) buffer ValueTmpBlock {uint ValueTmp[];};
layout (std430, binding = 27) buffer HistogramBlock {uint Histogram[];};
layout (std430, binding = 28) buffer PrefixBlock {uint PrefixSum[];};

uvec2 U2P(in uint64_t pckg) {
    //return uvec2(uint((pckg >> 0u) & 0xFFFFFFFFu), uint((pckg >> 32u) & 0xFFFFFFFFu));
    return unpackUint2x32(pckg);
}

struct blocks_info { uint count; uint offset; };
blocks_info get_blocks_info(in uint n) {
    uint block_count = n > 0 ? ((n - 1) / (BLOCK_SIZE * gl_NumWorkGroups.x) + 1) : 0;
    return blocks_info(block_count, gl_WorkGroupID.x * BLOCK_SIZE * block_count);
}

uint btc(in uint vlc){
    return vlc == 0 ? 0 : bitCount(vlc);
}

#ifdef EMULATE_BALLOT

shared uint ballotCache[WORK_SIZE];
shared uint invocationCache[WORK_SIZE][WARP_SIZE];

#define UVEC_BALLOT_WARP UVEC_WARP

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

uint readLane(in uint val, in int lane) {
    // warp can be have barrier, but is not required
    atomicExchange(invocationCache[LC_IDX][LANE_IDX], val);
    // warp can be have barrier, but is not required
    return invocationCache[LC_IDX][lane];
}

#else

//#define UVEC_BALLOT_WARP UVEC64_WARP
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
    return uint(btc(lh.x) + btc(lh.y));
}

uint readLane(in uint val, in int lane){
    return readInvocationARB(val, uint(lane));
}

//uint64_t readLane(in uint64_t val, in int lane){
//    return readInvocationARB(val, uint(lane));
//}

uvec2 ballotHW(in bool val) {
    return U2P(ballotARB(val));
}

#endif