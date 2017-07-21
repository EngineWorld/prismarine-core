
#include "./include/constants.hlsl"
#include "./include/structs.hlsl"
#include "./include/uniforms.hlsl"
#include "./include/STOmath.hlsl"
#include "./include/vertex.hlsl"





#include "./hlbvh/traverse.hlsl"

[numthreads(128, 1, 1)]
void main( uint3 WorkGroupID : SV_DispatchThreadID, uint3 LocalInvocationID : SV_GroupIndex )
{
    uint t = WorkGroupID.x * 128 + LocalInvocationID.x;
}