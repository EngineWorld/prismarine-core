#define EMULATE_BALLOT // compatible with SM5.0

#ifndef EMULATE_BALLOT
//#extension GL_ARB_gpu_shader_int64 : require
//#extension GL_ARB_shader_ballot : require
#endif

#define WORK_SIZE 32
#define WARP_SIZE 32
#define BLOCK_SIZE (WORK_SIZE*WARP_SIZE)
#define BITS_PER_PASS 4
#define RADICES 16
#define RADICES_MASK 0xf

#define WG_SIZE WORK_SIZE
#define WG_COUNT 8

static uint WG_IDX = 0;
static uint LC_IDX = 0;
static uint LANE_IDX = 0;

#define UVEC_WARP uint
#define BVEC_WARP bool
#define UVEC64_WARP uint64_t

//#define READ_LANE(V, I) ((I >= 0 && I < WARP_SIZE) ? readLane(V, I) : 0)
#define READ_LANE(V, I) (uint(I >= 0 && I < WARP_SIZE) * readLane(V, I))
#define BFE(m, i, s) ((m >> (i)) & ((1 << (s))-1))

RWStructuredBuffer<uint> KeyIn : register(u0);
RWStructuredBuffer<uint> ValueIn : register(u1);
RWStructuredBuffer<uint> KeyOut : register(u2);
RWStructuredBuffer<uint> ValueOut : register(u3);

struct VarsBlock {
    uint NumKeys;
    uint Shift;
    uint Descending;
    uint IsSigned;
};

RWStructuredBuffer<VarsBlock> vars : register(u4);
RWStructuredBuffer<uint> Histogram : register(u5);

struct blocks_info { uint count; uint offset; };
blocks_info get_blocks_info(in uint n, in uint wg_idx) {
     uint block_stride = WG_COUNT * BLOCK_SIZE;
     uint block_count = n > 0 ? (n - 1) / block_stride + 1 : 0;
     uint block_off = wg_idx * BLOCK_SIZE * block_count;
    blocks_info result;
    result.count = block_count;
    result.offset = block_off;
    return result;
}

#ifdef EMULATE_BALLOT

groupshared uint ballotCache[WORK_SIZE];
groupshared uint invocationCache[WORK_SIZE][WARP_SIZE];

#define UVEC_BALLOT_WARP UVEC_WARP

uint genLtMask(){
    return (1 << LANE_IDX)-1;
}

uint ballot(in bool val) {
    if (LANE_IDX == 0) ballotCache[LC_IDX] = 0;
    // warp can be have barrier, but is not required
    InterlockedOr(ballotCache[LC_IDX], uint(val) << LANE_IDX);
    // warp can be have barrier, but is not required
    return ballotCache[LC_IDX];
}

uint bitCount64(in uint a64) {
    return countbits(a64);
}

uint readLane(in uint val, in uint lane) {
    // warp can be have barrier, but is not required
    uint tmp = 0;
    InterlockedExchange(invocationCache[LC_IDX][LANE_IDX], val, tmp);
    // warp can be have barrier, but is not required
    return invocationCache[LC_IDX][lane];
}

#else

#define UVEC_BALLOT_WARP uint4

uint4 genLtMask(){
    if (LANE_IDX < 32 ) return uint4(                                    (1 << (LANE_IDX-0 ))-1, 0, 0, 0); else 
    if (LANE_IDX < 64 ) return uint4(0xFFFFFFFF,                         (1 << (LANE_IDX-32))-1, 0, 0   ); else 
    if (LANE_IDX < 96 ) return uint4(0xFFFFFFFF, 0xFFFFFFFF,             (1 << (LANE_IDX-64))-1, 0      ); else 
    if (LANE_IDX < 128) return uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, (1 << (LANE_IDX-96))-1         ); 
    return uint4(0,0,0,0);
}

uint bitCount64(in uint4 a64) {
    return countbits(a64.x) + countbits(a64.y) + countbits(a64.z) + countbits(a64.w);
}

uint readLane(in uint val, in uint lane){
    return WaveReadLaneAt(val, lane);
}

uint4 ballot(in bool val) {
    uint4 result = WaveActiveBallot(val);
    return result;
}

#endif