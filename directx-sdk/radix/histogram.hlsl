
#include "./includes.hlsl"

groupshared uint localHistogram[RADICES];

[numthreads(WARP_SIZE, WORK_SIZE, 1)]
void main( uint3 WorkGroupID : SV_DispatchThreadID, uint3 LocalInvocationID : SV_GroupID, uint3 GlobalInvocationID : SV_GroupThreadID, uint LocalInvocationIndex : SV_GroupIndex )
{
    WG_IDX = WorkGroupID.x;
    LC_IDX = LocalInvocationID.y;

#ifdef EMULATE_BALLOT
    LANE_IDX = LocalInvocationID.x;
#else
    LANE_IDX = WaveGetLaneIndex();
#endif

    blocks_info blocks = get_blocks_info(vars[0].NumKeys, WG_IDX);
    uint localIdx = WARP_SIZE * LC_IDX + LANE_IDX;
    if (localIdx < RADICES) localHistogram[localIdx] = 0;
    AllMemoryBarrierWithGroupSync();

    UVEC_WARP addr = blocks.offset + localIdx;
    for (int i=0;i<4096;i++) {
        if (i >= blocks.count) break;

         BVEC_WARP validAddress = addr < vars[0].NumKeys;
         UVEC_WARP data = KeyIn[addr];
         UVEC_WARP k = BFE(data, int(vars[0].Shift), BITS_PER_PASS);
         UVEC_WARP key = k;

        // smaller version (generalized)
        InterlockedAdd(localHistogram[key], UVEC_WARP(validAddress));
        addr += BLOCK_SIZE;
    }

    AllMemoryBarrierWithGroupSync();
    if (localIdx < RADICES) Histogram[localIdx * WG_COUNT + WG_IDX] = localHistogram[localIdx];
}
