
#include "./includes.hlsl"

groupshared uint localSort[BLOCK_SIZE];
groupshared uint localSortVal[BLOCK_SIZE];
groupshared uint localHistogramToCarry[RADICES];
groupshared uint localHistogram[RADICES];
groupshared uint warpPScan[WORK_SIZE];
groupshared uint psums[WORK_SIZE];
groupshared uint totals[WORK_SIZE];

// prefix scan for WARP vector
uint prefix_scan(
    inout UVEC_WARP v){
     BVEC_WARP btf = (v == 1);
     UVEC_BALLOT_WARP bits = ballot(btf);
     UVEC_WARP sum = bitCount64(bits);
    v = bitCount64(bits & genLtMask());
    return sum;
}


// WARP version of prefix_sum
uint prefix_sum(
    in uint data, inout uint total_sum) {
    UVEC_WARP rsort = data;
    for (uint i = 1; i < WORK_SIZE; i <<= 1) {
         UVEC_WARP tmp = READ_LANE(rsort, LANE_IDX - i) * uint(LANE_IDX < WORK_SIZE);
        rsort += tmp;
    }
    total_sum = READ_LANE(rsort, WORK_SIZE-1);
     UVEC_WARP result = READ_LANE(rsort, LANE_IDX - 1) * uint(LANE_IDX < WORK_SIZE);
    return result;
}



void sortBits(
    inout uint sort, inout uint sortVal){
     UVEC_WARP addr = WARP_SIZE * LC_IDX + LANE_IDX;
    for (int i=0;i<BITS_PER_PASS;i++) {
         BVEC_WARP cmp = BFE(sort, int(vars[0].Shift) + i, 1) == 0;
        UVEC_WARP warpKey = UVEC_WARP(cmp);

        // prefix scan for WARP vector
         uint pscan = prefix_scan(warpKey);
        if (LANE_IDX == 0) warpPScan[LC_IDX] = pscan; // cache by WARP ID
        AllMemoryBarrierWithGroupSync();

        // use LANE_IDX as LC_IDX, so we did caching
        if (LC_IDX == 0) {
            uint warpTotal = 0;
            uint prefixSum = prefix_sum(warpPScan[LANE_IDX], warpTotal);
            if (LANE_IDX < WORK_SIZE) {
                psums [LANE_IDX] = prefixSum;
                totals[LANE_IDX] = warpTotal;
            }
        }
        AllMemoryBarrierWithGroupSync();
        
        // use generalized local indexing (incl. warps)
        warpKey += psums[LC_IDX];

         UVEC_WARP destAddr = cmp ? warpKey : (addr - warpKey + totals[LC_IDX]);
        localSort   [destAddr] = sort;
        localSortVal[destAddr] = sortVal;
        AllMemoryBarrierWithGroupSync();

        sort    = localSort   [addr];
        sortVal = localSortVal[addr];
        AllMemoryBarrierWithGroupSync();
    }
}

[numthreads(WARP_SIZE, WORK_SIZE, 1)]
void CSMain( uint3 WorkGroupID : SV_DispatchThreadID, uint3 LocalInvocationID : SV_GroupID, uint3 GlobalInvocationID : SV_GroupThreadID, uint LocalInvocationIndex : SV_GroupIndex )
{
    WG_IDX = WorkGroupID.x;
    LC_IDX = LocalInvocationID.y;
    
#ifdef EMULATE_BALLOT
    LANE_IDX = LocalInvocationID.x;
#else
    LANE_IDX = WaveGetLaneIndex();
#endif

     UVEC_WARP localIdx = WARP_SIZE * LC_IDX + LANE_IDX;
     UVEC_WARP def = 0xFFFFFFFFu;

    if (localIdx < RADICES) localHistogramToCarry[localIdx] = Histogram[localIdx * WG_COUNT + WG_IDX];
    AllMemoryBarrierWithGroupSync();

    blocks_info blocks = get_blocks_info(vars[0].NumKeys, WG_IDX);
    UVEC_WARP addr = blocks.offset + localIdx;

    for ( int i_block=0; i_block < 4096 ; i_block++ ) {
        if (i_block >= blocks.count) break;

         BVEC_WARP validAddress = addr < vars[0].NumKeys;
        UVEC_WARP data      = validAddress ?   KeyIn[addr] : def;
        UVEC_WARP dataValue = validAddress ? ValueIn[addr] : 0u;
        sortBits(data, dataValue);

         UVEC_WARP k = BFE(data, int(vars[0].Shift), BITS_PER_PASS);
         UVEC_WARP      key = k;
         UVEC_WARP  histKey = key;
         UVEC_WARP localKey = key + (LC_IDX / RADICES) * RADICES;
        if (localIdx < WORK_SIZE) localSort[LANE_IDX] = 0;
        AllMemoryBarrierWithGroupSync();

        InterlockedAdd(localSort[localKey], uint(validAddress));
         UVEC_WARP offset = localHistogramToCarry[k] + localIdx;
        AllMemoryBarrierWithGroupSync();

        // Use LANE_IDX as LC_IDX (non-vector mode)
        if (LC_IDX == 0) {
            UVEC_WARP warpHistogram = 0;
            UVEC_WARP sum = 0;
            if (LANE_IDX < RADICES) {
                for (int i=0;i<WORK_SIZE/RADICES;i++) sum += localSort[i * RADICES + LANE_IDX];
                if (LANE_IDX < RADICES) localHistogramToCarry[LANE_IDX] += warpHistogram = sum;
            }

            uint tmp = 0;
            warpHistogram = READ_LANE(warpHistogram, LANE_IDX-1);
            tmp = READ_LANE(warpHistogram, LANE_IDX-3) + READ_LANE(warpHistogram, LANE_IDX-2) + READ_LANE(warpHistogram, LANE_IDX-1);
            warpHistogram += tmp;
            tmp = READ_LANE(warpHistogram, LANE_IDX-12) + READ_LANE(warpHistogram, LANE_IDX-8) + READ_LANE(warpHistogram, LANE_IDX-4);
            warpHistogram += tmp;
        
            if (LANE_IDX < RADICES) {
                localHistogram[LANE_IDX] = warpHistogram;
            }
        }
        AllMemoryBarrierWithGroupSync();

         UVEC_WARP outKey = offset - localHistogram[histKey];
        if (validAddress)   KeyOut[outKey] = data;
        if (validAddress) ValueOut[outKey] = dataValue;
        AllMemoryBarrierWithGroupSync();

        addr += BLOCK_SIZE;
    }
}

