#include "../include/constants.hlsl"
#include "../include/structs.hlsl"
#include "../include/uniforms.hlsl"
#include "./includes.hlsl"

[numthreads(WORK_SIZE, 1, 1)]
void CSMain( uint3 WorkGroupID : SV_GroupID, uint3 LocalInvocationID  : SV_GroupThreadID, uint3 GlobalInvocationID : SV_DispatchThreadID)
{
    uint globalID = WorkGroupID.x * WORK_SIZE + LocalInvocationID.x;
    if (globalID < geometryBlock[0].triangleCount) {
        int idx = Leafs[globalID].pdata.z;
        HlbvhNode nd = Nodes[idx];
        for(int i=0;i<256;i++) {
            idx = nd.pdata.z;
            if (idx <= 0) break; 

            nd = Nodes[idx];

            int tmp = 0;
            InterlockedCompareExchange(Flags[idx], 0, 1, tmp);
            if (tmp == 1) {
                HlbvhNode ln = Nodes[nd.pdata.x];
                HlbvhNode rn = Nodes[nd.pdata.y];
                bbox bound = nd.box;
                bound.mn = min(ln.box.mn, rn.box.mn) - 0.00001f;
                bound.mx = max(ln.box.mx, rn.box.mx) + 0.00001f;
                nd.box = bound;
                Nodes[idx] = nd;
            } else {
                break;
            }
        }
    }
}
