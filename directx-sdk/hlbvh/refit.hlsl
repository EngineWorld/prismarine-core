#include "../include/constants.hlsl"
#include "../include/structs.hlsl"
#include "../include/uniforms.hlsl"
#include "./includes.hlsl"

[numthreads(128, 1, 1)]
void main( uint3 WorkGroupID : SV_DispatchThreadID, uint3 LocalInvocationID : SV_GroupIndex )
{
    uint globalID = WorkGroupID.x * 128 + LocalInvocationID.x;
    if (globalID >= GEOMETRY_BLOCK geometryUniform.triangleCount) return;

    int idx = Leafs[globalID].pdata.z;
    HlbvhNode nd = Nodes[idx];
    for(int i=0;i<256;i++) {
        idx = nd.pdata.z;
        if (idx <= 0) break; 

        nd = Nodes[idx];

        int tmp = 0;
        InterlockedCompareExchange(Flags[idx], 0, 1, tmp);
        if (tmp) {
            HlbvhNode ln = Nodes[nd.pdata.x];
            HlbvhNode rn = Nodes[nd.pdata.y];
            Nodes[idx].box.mn = min(ln.box.mn, rn.box.mn) - 0.00001f;
            Nodes[idx].box.mx = max(ln.box.mx, rn.box.mx) + 0.00001f;
        } else {
            break;
        }
        
    }
}
