#include "../include/constants.hlsl"
#include "../include/structs.hlsl"
#include "../include/uniforms.hlsl"
#include "../include/vertex.hlsl"
#include "../include/morton.hlsl"
#include "../include/STOmath.hlsl"

RWStructuredBuffer<Leaf> OutLeafs : register(u0);
RWStructuredBuffer<uint> Mortoncodes : register(u1);
RWStructuredBuffer<int> MortoncodesIndices : register(u2);
RWStructuredBuffer<int> AABBCounter : register(u3);

uint add(inout uint mem, in uint ops){
    uint tmp = mem; mem += ops; return tmp;
}

[numthreads(WORK_SIZE, 1, 1)]
void CSMain( uint3 WorkGroupID : SV_GroupID, uint3 LocalInvocationID  : SV_GroupThreadID, uint3 GlobalInvocationID : SV_DispatchThreadID)
{
    uint t = WorkGroupID.x * WORK_SIZE + LocalInvocationID.x;
    if (t < geometryBlock[0].triangleCount) {

        float4 mn = float4(INFINITY, INFINITY, INFINITY, INFINITY);
        float4 mx = -mn;

        float3x4 triverts = float3x4(
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(t)), 0), 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(t)), 1), 
            fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(t)), 2)
        );

        triverts[0] = mul(geometryBlock[0].project, triverts[0]);
        triverts[1] = mul(geometryBlock[0].project, triverts[1]);
        triverts[2] = mul(geometryBlock[0].project, triverts[2]);

        //triverts[0] = mul(triverts[0], geometryBlock[0].project);
        //triverts[1] = mul(triverts[1], geometryBlock[0].project);
        //triverts[2] = mul(triverts[2], geometryBlock[0].project);

        float4 tcenter = (triverts[0] + triverts[1] + triverts[2]) * 0.33333333333333f;
        if (length(abs(triverts[0] - tcenter).xyz + abs(triverts[1] - tcenter).xyz + abs(triverts[2] - tcenter).xyz) < 1.e-5) return;

        mn = min(min(triverts[0], triverts[1]), triverts[2]);
        mx = max(max(triverts[0], triverts[1]), triverts[2]);

        // Starting from
        bbox stack[4];
        uint split[4];
        
        split[0] = 0;
        stack[0].mn = mn;
        stack[0].mx = mx;

        uint maxSplits = 0;
        uint countBox = 1;
        uint iteration = 0;
        uint i = 0;

        for (iteration=0;iteration<10;iteration++) {
            bbox current = stack[i];
            float4 center = (current.mn + current.mx) * 0.5f;

            if (split[i] >= maxSplits) {
                int to = 0;
                InterlockedAdd(AABBCounter[0], 1, to);

                MortoncodesIndices[to] = int(to);
                Mortoncodes[to] = encodeMorton3_64(clamp(
                    uint3(floor(clamp(center.xyz, (0.00001f).xxx, (0.99999f).xxx) * 1024.0f)), 
                    (0).xxx, (1023).xxx));

                Leaf outLeaf = OutLeafs[to];
                outLeaf.box.mn = current.mn - 0.0001f;
                outLeaf.box.mx = current.mx + 0.0001f;
                outLeaf.pdata.xy = int2(to, to);
                outLeaf.pdata.zw = int2(-1, t);
                OutLeafs[to] = outLeaf;
            } else {
                float4 diff = current.mx - current.mn;
                uint halfBoxLeft  = add(countBox, 2);
                uint halfBoxRight = halfBoxLeft+1;
                float longest = max(max(diff.x, diff.y), diff.z);

                bbox leftBox  = current;
                bbox rightBox = current;
                if (equalF(longest, diff.x)) {
                    rightBox.mn.x = leftBox.mx.x = center.x;
                } else 
                if (equalF(longest, diff.y)) {
                    rightBox.mn.y = leftBox.mx.y = center.y;
                } else {
                    rightBox.mn.z = leftBox.mx.z = center.z;
                }
                
                split[halfBoxLeft] = split[i]+1;
                stack[halfBoxLeft] = leftBox;
                split[halfBoxRight] = split[i]+1;
                stack[halfBoxRight] = rightBox;
            }

            i++;
        }
    }
}
