#include "../include/constants.hlsl"
#include "../include/structs.hlsl"
#include "../include/uniforms.hlsl"
#include "../include/vertex.hlsl"
#include "../include/STOmath.hlsl"

#define WARP_SIZE 32

RWStructuredBuffer<float4> minmax : register(u5);

groupshared bbox sdata[ 512 ];

bbox getMinMaxPrimitive(in uint idx){
     uint tri = clamp(idx, 0u, uint(GEOMETRY_BLOCK geometryUniform.triangleCount-1));

    float3x4 triverts = float3x4(
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 0), 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 1), 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 2)
    );

    triverts[0] = mul(triverts[0], GEOMETRY_BLOCK octreeUniform.project);
    triverts[1] = mul(triverts[1], GEOMETRY_BLOCK octreeUniform.project);
    triverts[2] = mul(triverts[2], GEOMETRY_BLOCK octreeUniform.project);

    bbox result;
    result.mn = min(min(triverts[0], triverts[1]), triverts[2]) - float4(0.00001f, 0.00001f, 0.00001f, 0.00001f);
    result.mx = max(max(triverts[0], triverts[1]), triverts[2]) + float4(0.00001f, 0.00001f, 0.00001f, 0.00001f);
    return result;
}

bbox bboxunion(in bbox b1, in bbox b2) {
    bbox result;
    result.mn = min(b1.mn, b2.mn);
    result.mx = max(b1.mx, b2.mx);
    return result;
}

[numthreads(512, 1, 1)]
void main( uint3 WorkGroupID : SV_DispatchThreadID, uint3 LocalInvocationID : SV_GroupIndex )
{
    uint tid = LocalInvocationID.x;
    uint gridSize = (512*2)*1;
    uint tcount = min(GEOMETRY_BLOCK geometryUniform.triangleCount, 16777216);
    uint i = WorkGroupID.x * (512*2) + tid;

    sdata[tid].mn =  float4(100000.f, 100000.f, 100000.f, 100000.f);
    sdata[tid].mx = -float4(100000.f, 100000.f, 100000.f, 100000.f);
    while (i < tcount) {
        sdata[tid] = bboxunion(sdata[tid], bboxunion(getMinMaxPrimitive(i), getMinMaxPrimitive(i + 512)));
        i += gridSize;
    };
    AllMemoryBarrierWithGroupSync();

    for (uint ik=(512>>1);ik>=1;ik>>=1) {
        if (tid < ik) {
            sdata[tid] = bboxunion(sdata[tid], sdata[tid + ik]); 
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
