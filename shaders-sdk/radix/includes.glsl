//#define EMULATE_BALLOT

#ifndef EMULATE_BALLOT
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_ARB_shader_ballot : require
#endif

#define WORK_SIZE 32
#define WARP_SIZE 32
#define BLOCK_SIZE (WORK_SIZE*WARP_SIZE)
#define BITS_PER_PASS 4
#define RADICES 16
#define RADICES_MASK 0xf

#define WG_SIZE WORK_SIZE
#define WG_COUNT 8
#define WG_IDX gl_WorkGroupID.x
#define LC_IDX gl_LocalInvocationID.y
#define LANE_IDX gl_LocalInvocationID.x

#define UVEC_WARP uint
#define BVEC_WARP bool
#define UVEC64_WARP uint64_t
#define KEYTYPE UVEC64_WARP

//#define READ_LANE(V, I) ((I >= 0 && I < WARP_SIZE) ? readLane(V, I) : 0)
#define READ_LANE(V, I) (uint(I >= 0 && I < WARP_SIZE) * readLane(V, I))

#define BFE(a,o,n) ((a >> o) & ((1 << n)-1))

layout (std430, binding = 20) restrict buffer KeyInBlock {KEYTYPE KeyIn[];};
layout (std430, binding = 21) restrict buffer ValueInBlock {uint ValueIn[];};
layout (std430, binding = 22) restrict buffer KeyOutBlock {KEYTYPE KeyOut[];};
layout (std430, binding = 23) restrict buffer ValueOutBlock {uint ValueOut[];};
layout (std430, binding = 24) restrict buffer VarsBlock {
    uint NumKeys;
    uint Shift;
    uint Descending;
    uint IsSigned;
};
layout (std430, binding = 25) restrict buffer HistogramBlock {uint Histogram[];};

struct blocks_info { uint count; uint offset; };
blocks_info get_blocks_info(in uint n, in uint wg_idx) {
    uint block_stride = WG_COUNT * BLOCK_SIZE;
    uint block_count = n > 0 ? (n - 1) / block_stride + 1 : 0;
    uint block_off = wg_idx * BLOCK_SIZE * block_count;
    return blocks_info(block_count, block_off);
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

uint readLane(in uint val, in uint lane) {
    // warp can be have barrier, but is not required
    atomicExchange(invocationCache[LC_IDX][LANE_IDX], val);
    // warp can be have barrier, but is not required
    return invocationCache[LC_IDX][lane];
}

#else

//#define UVEC_BALLOT_WARP UVEC64_WARP
#define UVEC_BALLOT_WARP uvec2

uvec2 genLtMask(){
    return unpackUint2x32(gl_SubGroupLtMaskARB);
}

uint bitCount64(in uvec2 lh) {
    return uint(btc(lh.x) + btc(lh.y));
}

uint readLane(in uint val, in uint lane){
    return readInvocationARB(val, lane);
}

uvec2 ballotHW(in bool val) {
    return unpackUint2x32(ballotARB(val)) & uvec2(
        gl_SubGroupSizeARB >= 32 ? 0xFFFFFFFF : ((1 << gl_SubGroupSizeARB)-1), 
        gl_SubGroupSizeARB >= 64 ? 0xFFFFFFFF : ((1 << (gl_SubGroupSizeARB-32))-1)
    );
}

#endif