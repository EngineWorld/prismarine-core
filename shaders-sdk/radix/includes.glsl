
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_ARB_shader_ballot : require

#define WORK_SIZE 32
#define WARP_SIZE 32
#define BLOCK_SIZE (WORK_SIZE*WARP_SIZE)
#define BITS_PER_PASS 4
#define RADICES 16
#define RADICES_MASK 0xf

#define WG_SIZE WORK_SIZE
#define WG_COUNT 64
#define WG_IDX gl_WorkGroupID.x
#define LC_IDX gl_LocalInvocationID.y
#define LANE_IDX gl_SubGroupInvocationARB

#define UVEC_WARP uint
#define BVEC_WARP bool
#define UVEC64_WARP uint64_t

//#define READ_LANE(V, I) ((I >= 0 && I < WARP_SIZE) ? readInvocationARB(V, I) : 0)
#define READ_LANE(V, I) (uint(I >= 0 && I < WARP_SIZE) * readInvocationARB(V, I))

layout (std430, binding = 0) buffer KeyInBlock {uint KeyIn[];};
layout (std430, binding = 1) buffer ValueInBlock {uint ValueIn[];};
layout (std430, binding = 2) buffer KeyOutBlock {uint KeyOut[];};
layout (std430, binding = 3) buffer ValueOutBlock {uint ValueOut[];};
layout (std430, binding = 4) buffer VarsBlock {
    uint NumKeys;
    uint Shift;
    uint Descending;
    uint IsSigned;
};
layout (std430, binding = 5) buffer HistogramBlock {uint Histogram[];};

struct blocks_info { uint count; uint offset; };
blocks_info get_blocks_info(const uint n, const uint wg_idx) {
    /*
    const uint aligned = n + BLOCK_SIZE - (n % BLOCK_SIZE);
    const uint blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const uint blocks_per_wg = (blocks + WG_COUNT - 1) / WG_COUNT;
    const int n_blocks = int(aligned / BLOCK_SIZE) - int(blocks_per_wg * wg_idx);
    return blocks_info(uint(clamp(n_blocks, 0, int(blocks_per_wg))), blocks_per_wg * BLOCK_SIZE * wg_idx);
    */

    const uint block_stride = WG_COUNT * BLOCK_SIZE;
    const uint block_count = uint(ceil(double(n) / double(block_stride)));
    const uint block_off = wg_idx * BLOCK_SIZE * block_count;
    return blocks_info(block_count, block_off);
}

