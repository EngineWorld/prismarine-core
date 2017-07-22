
#include "./include/constants.hlsl"
#include "./include/structs.hlsl"
#include "./include/uniforms.hlsl"
#include "./include/STOmath.hlsl"
#include "./include/vertex.hlsl"





#include "./hlbvh/traverse.hlsl"

[numthreads(WORK_SIZE, 1, 1)]
void main( uint3 WorkGroupID : SV_DispatchThreadID, uint3 LocalInvocationID : SV_GroupID, uint3 GlobalInvocationID : SV_GroupThreadID, uint LocalInvocationIndex : SV_GroupIndex )
{
    uint t = WorkGroupID.x * WORK_SIZE + LocalInvocationID.x;
}