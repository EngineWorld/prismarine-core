
#include "../include/constants.hlsl"
#include "../include/structs.hlsl"
#include "../include/uniforms.hlsl"
#include "../include/STOmath.hlsl"
#include "../include/vertex.hlsl"
#include "../include/rays.hlsl"
#include "../hlbvh/traverse.hlsl"



struct LResult {
    float4 normal;
    float4 tangent;
    float4 texcoord;
    float4 color;
    float4 mods;

    float dist;
    int triangleID;
    int materialID;
    float predist; // legacy
    float4 uv; // legacy
};

LResult loadInfo(in TResult hitp) {
    LResult res;
    res.dist = hitp.dist;
    res.triangleID = hitp.triangleID;
    res.materialID = -1;
    res.predist = hitp.predist; // legacy
    res.uv = hitp.uv; // legacy

    int tri = hitp.triangleID;
    if (greaterEqualF(hitp.dist, 0.0f) && lessF(hitp.dist, INFINITY) && tri != LONGEST) {
        float3 trinorms[3];
        float3 triverts[3];
        float4 texcoords[3];
        float4 colors[3];
        float4 mods[3];

        for (int x=0;x<3;x++) {
            int2 mos = gatherMosaic(getUniformCoord(tri));
            triverts[x] = fetchMosaic(vertex_texture, mos, x).xyz;
            trinorms[x] = fetchMosaic(normal_texture, mos, x).xyz;
            texcoords[x] = fetchMosaic(texcoords_texture, mos, x);
            colors[x] = (1.0f).xxxx;
            mods[x] = fetchMosaic(modifiers_texture, mos, x);
        }

        float3 deltaPos1 = triverts[1] - triverts[0];
        float3 deltaPos2 = triverts[2] - triverts[0];
        float2 uv = hitp.uv.xy;
        float3 nor = normalize(cross(deltaPos1, deltaPos2));
        float3 normal = mad(trinorms[0], (1.0f - uv.x - uv.y).xxx, mad(trinorms[1], (uv.x).xxx, trinorms[2] * (uv.y).xxx));
        normal = lessF(length(normal), 0.f) ? nor : normalize(normal);
        normal = normal * sign(dot(normal, nor));

        bool delta = all((texcoords[0].xy == texcoords[1].xy)) && all((texcoords[0].xy == texcoords[2].xy));
        float2 deltaUV1 = delta ? float2(1.0f, 0.0f) : texcoords[1].xy - texcoords[0].xy;
        float2 deltaUV2 = delta ? float2(0.0f, 1.0f) : texcoords[2].xy - texcoords[0].xy;

        float f = 1.0f / mad(deltaUV1.x, deltaUV2.y, -deltaUV1.y * deltaUV2.x);
        float3 tan = mad(deltaPos1, (deltaUV2.y).xxx, -deltaPos2 * deltaUV1.y) * f;

        res.normal   = float4(normal, 0.0f);
        res.tangent  = float4(normalize(tan - normal * sign(dot(tan, nor))), 0.0f);
        res.texcoord = mad(texcoords[0], (1.0f - uv.x - uv.y).xxxx, mad(texcoords[1], (uv.x).xxxx, texcoords[2] * (uv.y).xxxx));
        res.color    = mad(   colors[0], (1.0f - uv.x - uv.y).xxxx, mad(   colors[1], (uv.x).xxxx,    colors[2] * (uv.y).xxxx));
        res.mods     = mad(     mods[0], (1.0f - uv.x - uv.y).xxxx, mad(     mods[1], (uv.x).xxxx,      mods[2] * (uv.y).xxxx));
        res.materialID = mats[tri];
    }

    return res;
}


[numthreads(WORK_SIZE, 1, 1)]
void CSMain( uint3 WorkGroupID : SV_GroupID, uint3 LocalInvocationID  : SV_GroupThreadID, uint3 GlobalInvocationID : SV_DispatchThreadID)
{
    uint it = GlobalInvocationID.x;
    bool overflow = it >= rayBlock[0].rayCount;
    if (overflow) return;

    int t = activedBuf[it];
    Ray ray = fetchRayDirect(t);
    if (ray.actived < 1 || overflow) return;

    Hit hit = fetchHitDirect(t);

    LResult res = loadInfo(traverse(LocalInvocationID.x, hit.dist, ray.origin.xyz, ray.direct.xyz, int(floor(1.f + hit.vmods.w))));

    if (
        greaterEqualF(res.dist, 0.0f) &&
        lessF(res.dist, INFINITY) &&

        (lessEqualF(res.dist, hit.dist) || geometryBlock[0].clearDepth > 0) &&
        
        res.materialID >= 0 &&
        res.materialID != LONGEST
    ) {
        Hit newHit = hit;
        float4 sysmod = newHit.vmods;

        newHit.normal = res.normal;
        newHit.tangent = res.tangent;
        newHit.texcoord = res.texcoord;
        newHit.vcolor = res.color;
        newHit.vmods = res.mods;
        newHit.dist = res.dist;
        newHit.triangleID = res.triangleID;
        newHit.materialID = res.materialID;
        newHit.shaded = 0;
        newHit.vmods.w = sysmod.w;

        if (!overflow) storeHit(t, newHit);
    }

}