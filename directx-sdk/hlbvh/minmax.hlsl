#include "../include/constants.hlsl"
#include "../include/structs.hlsl"
#include "../include/uniforms.hlsl"
#include "../include/vertex.hlsl"
#include "../include/STOmath.hlsl"

#define WARP_SIZE 32
#define LOCAL_SIZE 512
#define WORK_COUNT 1

RWStructuredBuffer<float4> minmax : register(u5);

groupshared bbox sdata[ LOCAL_SIZE ];

bbox getMinMaxPrimitive(in uint idx){
     uint tri = clamp(idx, 0u, uint(geometryBlock[0].triangleCount-1));

    float3x4 triverts = float3x4(
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 0), 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 1), 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 2)
    );

    triverts[0] = mul(triverts[0], geometryBlock[0].project);
    triverts[1] = mul(triverts[1], geometryBlock[0].project);
    triverts[2] = mul(triverts[2], geometryBlock[0].project);

    bbox result;
    result.mn = min(min(triverts[0], triverts[1]), triverts[2]) - (0.00001f).xxxx;
    result.mx = max(max(triverts[0], triverts[1]), triverts[2]) + (0.00001f).xxxx;
    return result;
}

void bboxunion(inout bbox result, in bbox b1, in bbox b2) {
    result.mn = min(b1.mn, b2.mn);
    result.mx = max(b1.mx, b2.mx);
}

bbox bboxunion(in bbox b1, in bbox b2) {
    bbox result;
    bboxunion(result, b1, b2);
    return result;
}

[numthreads(LOCAL_SIZE, 1, 1)]
void CSMain( uint3 WorkGroupID : SV_DispatchThreadID, uint3 LocalInvocationID : SV_GroupID, uint3 GlobalInvocationID : SV_GroupThreadID, uint LocalInvocationIndex : SV_GroupIndex )
{
    uint tid = LocalInvocationID.x;
    uint gridSize = (LOCAL_SIZE*2)*WORK_COUNT;
    uint tcount = min(geometryBlock[0].triangleCount, 16777216);
    uint i = WorkGroupID.x * (LOCAL_SIZE*2) + tid;

    bbox initial;
    initial.mn = (100000.f).xxxx;
    initial.mx = -initial.mn;
    sdata[tid] = initial;

    while (i < tcount) {
        bbox bound = sdata[tid];
        sdata[tid] = bboxunion(bound, bboxunion(getMinMaxPrimitive(i), getMinMaxPrimitive(i + 512)));
        i += gridSize;
    };
    AllMemoryBarrierWithGroupSync();

    for (uint ik=(LOCAL_SIZE>>1);ik>=1;ik>>=1) {
        if (tid < ik) {
            bbox bound = sdata[tid];
            bbox opbound = sdata[tid + ik];
            sdata[tid] = bboxunion(bound, opbound);
        }
        if (ik > WARP_SIZE) {
            AllMemoryBarrierWithGroupSync();
        }
    }

    if (tid == 0) {
        minmax[WorkGroupID.x*2 + 0] = sdata[0].mn;
        minmax[WorkGroupID.x*2 + 1] = sdata[0].mx;
    }
}
