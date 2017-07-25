
struct TResult {
    float dist;
    int triangleID;
    int materialID;
    float predist;
    float4 uv;
};

RWStructuredBuffer<HlbvhNode> Nodes : register(u9);

static const int bakedFragments = 8;
groupshared int bakedStack[WORK_SIZE][bakedFragments];

struct SharedVarData {
    float4 bakedRangeIntersection;
    int bakedRange;
    int bakedStackCount;
    uint L;
};

// WARP optimized triangle intersection
float intersectTriangle(in float3 orig, in float3 dir, in float3x3 ve, inout float2 UV, in bool valid) {
    if (!valid) return INFINITY;

    float3 e1 = ve[1] - ve[0];
    float3 e2 = ve[2] - ve[0];

    valid = valid && !(length(e1) < 0.00001f && length(e2) < 0.00001f);
    if (!valid) return INFINITY;

    float3 pvec = cross(dir, e2);
    float det = dot(e1, pvec);

#ifndef CULLING
    if (abs(det) <= 0.0f) valid = false;
#else
    if (det <= 0.0f) valid = false;
#endif
    if (!valid) return INFINITY;

    float3 tvec = orig - ve[0];
    float u = dot(tvec, pvec);
    float3 qvec = cross(tvec, e1);
    float v = dot(dir, qvec);
    float3 uvt = float3(u, v, dot(e2, qvec)) / det;

    if (
        any( uvt.xy                       < (0.f).xx) || 
        any((uvt.xx + float2(0.f, uvt.y)) > (1.f).xx) 
    ) valid = false;
    if (!valid) return INFINITY;

    UV.xy = uvt.xy;
    return (lessF(uvt.z, 0.0f) || !valid) ? INFINITY : uvt.z;
}

TResult choiceFirstBaked(inout SharedVarData sharedVarData, inout TResult res) {
     int tri = exchange(sharedVarData.bakedRange, int(LONGEST));

     bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;
        
    if (!validTriangle) return res;

     float2 uv = sharedVarData.bakedRangeIntersection.yz;
     float _d = sharedVarData.bakedRangeIntersection.x;
     bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangleID = tri;
        res.uv.xy = uv.xy;
    }

    return res;
}

void reorderTriangles(inout SharedVarData sharedVarData) {
    sharedVarData.bakedStackCount = min(sharedVarData.bakedStackCount, bakedFragments);
    int roundi = 0;
    for (roundi = 1; roundi < sharedVarData.bakedStackCount; roundi++) {
        for (int index = 0; index < sharedVarData.bakedStackCount - roundi; index++) {
            if (bakedStack[sharedVarData.L][index] <= bakedStack[sharedVarData.L][index+1]) {
                swap(bakedStack[sharedVarData.L][index], bakedStack[sharedVarData.L][index+1]);
            }
        }
    }

    // clean from dublicates
    int cleanBakedStack[bakedFragments];
    int cleanBakedStackCount = 0;
    cleanBakedStack[cleanBakedStackCount++] = bakedStack[sharedVarData.L][0];
    for (roundi = 1; roundi < sharedVarData.bakedStackCount; roundi++) {
        if(bakedStack[sharedVarData.L][roundi-1] != bakedStack[sharedVarData.L][roundi]) {
            cleanBakedStack[cleanBakedStackCount++] = bakedStack[sharedVarData.L][roundi];
        }
    }

    for (int i=0;i<sharedVarData.bakedStackCount;i++) bakedStack[sharedVarData.L][i] = cleanBakedStack[i];
    sharedVarData.bakedStackCount = cleanBakedStackCount;
}

TResult choiceBaked(inout SharedVarData sharedVarData, inout TResult res, in float3 orig, in float3 dir, in int tpi) {
    choiceFirstBaked(sharedVarData, res);
    reorderTriangles(sharedVarData);

    int tri = (tpi < exchange(sharedVarData.bakedStackCount, 0)) ? bakedStack[sharedVarData.L][tpi] : LONGEST;

    bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;

    // fetch directly
    float3x3 triverts = float3x3(
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 0).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 1).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 2).xyz
    );

    float2 uv = (0.0f).xx;
    float _d = intersectTriangle(orig, dir, triverts, uv, validTriangle);
    bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangleID = tri;
        res.uv.xy = uv.xy;
    }
    
    return res;
}

TResult testIntersection(inout SharedVarData sharedVarData, inout TResult res, in float3 orig, in float3 dir, in int tri, in bool isValid) {
    bool validTriangle = 
        isValid && 
        tri >= 0 && 
        tri != res.triangleID &&
        tri != LONGEST;

    float3x3 triverts = float3x3(
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 0).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 1).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 2).xyz
    );

    float2 uv = (0.0f).xx;
    float _d = intersectTriangle(orig, dir, triverts, uv, validTriangle);
    bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.predist) && greaterEqualF(_d, 0.0f);
    bool inbaked = equalF(_d, 0.0f);
    bool isbaked = equalF(_d, res.predist);
    bool changed = !isbaked && !inbaked;

    if (near) {
        if ( changed ) {
            res.predist = _d;
            res.triangleID = tri;
        }
        if ( inbaked ) {
            bakedStack[sharedVarData.L][sharedVarData.bakedStackCount++] = tri;
        } else 
        if (sharedVarData.bakedRange < tri || sharedVarData.bakedRange == LONGEST || changed ) {
            sharedVarData.bakedRange = tri;
            sharedVarData.bakedRangeIntersection = float4(_d, uv, 0.f);
        }
    }

    return res;
}

float3 projectVoxels(in float3 orig) {
    float4 nps = mul(geometryBlock[0].project, float4(orig, 1.0f));
    return nps.xyz / nps.w;
}

float3 unprojectVoxels(in float3 orig) {
    float4 nps = mul(geometryBlock[0].unproject, float4(orig, 1.0f));
    return nps.xyz / nps.w;
}

float intersectCubeSingle(in float3 origin, in float3 ray, in float4 cubeMin, in float4 cubeMax, inout float near, inout float far) {
     float3 dr = 1.0f / ray;
     float3 tMin = (cubeMin.xyz - origin) * dr;
     float3 tMax = (cubeMax.xyz - origin) * dr;
     float3 t1 = min(tMin, tMax);
     float3 t2 = max(tMin, tMax);
#ifdef ENABLE_AMD_INSTRUCTION_SET
     float tNear = max3(t1.x, t1.y, t1.z);
     float tFar  = min3(t2.x, t2.y, t2.z);
#else
     float tNear = max(max(t1.x, t1.y), t1.z);
     float tFar  = min(min(t2.x, t2.y), t2.z);
#endif
     bool isCube = tFar >= tNear && greaterEqualF(tFar, 0.0f);
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}


void intersectCubeApart(in float3 origin, in float3 ray, in float4 cubeMin, in float4 cubeMax, inout float near, inout float far) {
     float3 dr = 1.0f / ray;
     float3 tMin = (cubeMin.xyz - origin) * dr;
     float3 tMax = (cubeMax.xyz - origin) * dr;
     float3 t1 = min(tMin, tMax);
     float3 t2 = max(tMin, tMax);
#ifdef ENABLE_AMD_INSTRUCTION_SET
    near = max3(t1.x, t1.y, t1.z);
    far  = min3(t2.x, t2.y, t2.z);
#else
    near = max(max(t1.x, t1.y), t1.z);
    far  = min(min(t2.x, t2.y), t2.z);
#endif
}



static const float3 padding = (0.00001f).xxx;
static const int STACK_SIZE = 16;
groupshared int deferredStack[WORK_SIZE][STACK_SIZE];

bool2 and2(in bool2 a, in bool2 b){
    return bool2(a.x && b.x, a.y && b.y);
}

bool2 or2(in bool2 a, in bool2 b){
    return bool2(a.x || b.x, a.y || b.y);
}

bool2 not2(in bool2 a){
    return bool2(!a.x, !a.y);
}

TResult traverse(in uint L, in float distn, in float3 origin, in float3 direct, in int bakedStep) {
    TResult lastRes;
    lastRes.dist = INFINITY;
    lastRes.predist = INFINITY;
    lastRes.triangleID = LONGEST;
    lastRes.materialID = LONGEST;
    
    deferredStack[L][0] = -1;
    SharedVarData sharedVarData;
    sharedVarData.bakedStackCount = 0;
    sharedVarData.bakedRange = LONGEST;
    sharedVarData.L = L;

    int idx = 0, deferredPtr = 0;
    bool validBox = false;
    bool skip = false;

    // test constants
    float3 torig = projectVoxels(origin);
    float3 tdirproj = mul(geometryBlock[0].project, float4(direct, 0.0)).xyz;
    float dirlen = 1.0f / length(tdirproj);
    float3 dirproj = normalize(tdirproj);

    // test with root node
    float near = INFINITY, far = INFINITY;
    float d = intersectCubeSingle(torig, dirproj, float4((0.0f).xxx, 1.0f), float4((1.0f).xxx, 1.0f), near, far);
    lastRes.predist = far * dirlen;

    // init state
    {
        validBox = lessF(d, INFINITY) && greaterEqualF(d, 0.0f);
    }

    for(int i=0;i<8192;i++) {
        if (!validBox) break;
        HlbvhNode node = Nodes[idx];

        // is leaf
         bool isLeaf = node.pdata.x == node.pdata.y && validBox;
        if (isLeaf) {
            testIntersection(sharedVarData, lastRes, origin, direct, node.pdata.w, isLeaf);
        }

        bool notLeaf = node.pdata.x != node.pdata.y && validBox;
        if (notLeaf) {
            HlbvhNode lnode = Nodes[node.pdata.x];
            HlbvhNode rnode = Nodes[node.pdata.y];

            float2 inf2 = float2(INFINITY, INFINITY);
            float2 nearsLR = inf2;
            float2 farsLR = inf2;
            intersectCubeApart(torig, dirproj, lnode.box.mn, lnode.box.mx, nearsLR.x, farsLR.x);
            intersectCubeApart(torig, dirproj, rnode.box.mn, rnode.box.mx, nearsLR.y, farsLR.y);

            bool2 isCube = and2(farsLR >= nearsLR, farsLR >= (0.f).xx);
             
            //float2 nears = lerp(inf2, nearsLR, isCube);
            //float2  fars = lerp(inf2, farsLR, isCube);
            //float2  hits = lerp(nears, fars, nears < (0.f).xx);

            float2 nears = float2(isCube.x ? nearsLR.x : INFINITY, isCube.y ? nearsLR.y : INFINITY);
            float2  fars = float2(isCube.x ?  farsLR.x : INFINITY, isCube.y ?  farsLR.y : INFINITY);
            float2  hits = float2(nears.x < 0.0f ? fars.x : nears.x, nears.y < 0.0f ? fars.y : nears.y);

            bool2 overlaps = 
                and2(bool2(notLeaf, notLeaf), 
                and2((hits <= float2(INFINITY-PZERO, INFINITY-PZERO)),
                and2((hits > -float2(PZERO, PZERO)),
                (float2(lastRes.predist, lastRes.predist) > nears * dirlen - PZERO))));
            
            bool anyOverlap = any(overlaps);
            if (anyOverlap) {
                 bool leftOrder = all(overlaps) ? lessEqualF(hits.x, hits.y) : overlaps.x;

                //int2 leftright = overlaps ? node.pdata.xy : int2(-1, -1); // XShaderCompiler have bug with
                int2 leftright = int2(
                    overlaps.x ? node.pdata.x : -1,
                    overlaps.y ? node.pdata.y : -1
                );
                leftright = leftOrder ? leftright : leftright.yx;

                if (anyOverlap) {
                    if (deferredPtr < STACK_SIZE && leftright.y != -1) {
                        deferredStack[L][deferredPtr++] = leftright.y;
                    }
                    idx = leftright.x;
                    skip = true;
                }
            }
        }

        if (!skip) {
            int ptr = --deferredPtr;
            bool valid = ptr >= 0;
            idx = -1;
            idx = valid ? exchange(deferredStack[L][ptr], -1) : -1;
            validBox = validBox && valid && idx >= 0;
        } skip = false;
    }

    choiceBaked(sharedVarData, lastRes, origin, direct, bakedStep);
    return lastRes;
}
