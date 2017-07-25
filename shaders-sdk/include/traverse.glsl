layout ( std430, binding = 9 ) readonly buffer NodesBlock { HlbvhNode Nodes[]; };

//vec4 bakedRangeIntersection[1];
//int bakedRange[1];

const int bakedFragments = 8;
shared int bakedStack[WORK_SIZE][bakedFragments];
//int bakedStackCount = 0;

struct SharedVarsData {
    vec4 bakedRangeIntersection;
    int bakedRange;
    int bakedStackCount;
    uint L;
};

struct TResult {
    float dist;
    int triangle;
    int materialID; // legacy
    float predist;
    vec4 uv;
};

// WARP optimized triangle intersection
float intersectTriangle(in vec3 orig, in vec3 dir, in mat3 ve, inout vec2 UV, in bool valid) {
    //if (allInvocationsARB(!valid)) return INFINITY;
    if (!valid) return INFINITY;

     vec3 e1 = ve[1] - ve[0];
     vec3 e2 = ve[2] - ve[0];

    valid = valid && !(length(e1) < 0.00001f && length(e2) < 0.00001f);
    //if (allInvocationsARB(!valid)) return INFINITY;
    if (!valid) return INFINITY;

     vec3 pvec = cross(dir, e2);
     float det = dot(e1, pvec);

#ifndef CULLING
    if (abs(det) <= 0.0f) valid = false;
#else
    if (det <= 0.0f) valid = false;
#endif
    //if (allInvocationsARB(!valid)) return INFINITY;
    if (!valid) return INFINITY;

     vec3 tvec = orig - ve[0];
     float u = dot(tvec, pvec);
     vec3 qvec = cross(tvec, e1);
     float v = dot(dir, qvec);
     vec3 uvt = vec3(u, v, dot(e2, qvec)) / det;

    if (
        any(lessThan(uvt.xy, vec2(0.f))) || 
        any(greaterThan(vec2(uvt.x) + vec2(0.f, uvt.y), vec2(1.f))) 
    ) valid = false;
    //if (allInvocationsARB(!valid)) return INFINITY;
    if (!valid) return INFINITY;

    UV.xy = uvt.xy;
    return (lessF(uvt.z, 0.0f) || !valid) ? INFINITY : uvt.z;
}

TResult choiceFirstBaked(inout SharedVarsData sharedVarsData, inout TResult res) {
     int tri = exchange(sharedVarsData.bakedRange, LONGEST);

     bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;
        
    //if (allInvocationsARB(!validTriangle)) return res;
    if (!validTriangle) return res;

     vec2 uv = sharedVarsData.bakedRangeIntersection.yz;
     float _d = sharedVarsData.bakedRangeIntersection.x;
     bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv.xy;
    }

    return res;
}

void reorderTriangles(inout SharedVarsData sharedVarsData) {
    sharedVarsData.bakedStackCount = min(sharedVarsData.bakedStackCount, bakedFragments);
    for (int iround = 1; iround < sharedVarsData.bakedStackCount; iround++) {
        for (int index = 0; index < sharedVarsData.bakedStackCount - iround; index++) {
            if (bakedStack[sharedVarsData.L][index] <= bakedStack[sharedVarsData.L][index+1]) {
                swap(bakedStack[sharedVarsData.L][index], bakedStack[sharedVarsData.L][index+1]);
            }
        }
    }

    // clean from dublicates
    int cleanBakedStack[bakedFragments];
    int cleanBakedStackCount = 0;
    cleanBakedStack[cleanBakedStackCount++] = bakedStack[sharedVarsData.L][0];
    for (int iround = 1; iround < sharedVarsData.bakedStackCount; iround++) {
        if(bakedStack[sharedVarsData.L][iround-1] != bakedStack[sharedVarsData.L][iround]) {
            cleanBakedStack[cleanBakedStackCount++] = bakedStack[sharedVarsData.L][iround];
        }
    }

    for (int i=0;i<sharedVarsData.bakedStackCount;i++) bakedStack[sharedVarsData.L][i] = cleanBakedStack[i];
    sharedVarsData.bakedStackCount = cleanBakedStackCount;
}

TResult choiceBaked(inout SharedVarsData sharedVarsData, inout TResult res, in vec3 orig, in vec3 dir, in int tpi) {
    choiceFirstBaked(sharedVarsData, res);
    reorderTriangles(sharedVarsData);

     int tri = tpi < exchange(sharedVarsData.bakedStackCount, 0) ? bakedStack[sharedVarsData.L][tpi] : LONGEST;

     bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;

    // fetch directly
    mat3 triverts = mat3(
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 0).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 1).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 2).xyz
    );

    vec2 uv = vec2(0.0f);
     float _d = intersectTriangle(orig, dir, triverts, uv, validTriangle);
     bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv.xy;
    }
    
    return res;
}

TResult testIntersection(inout SharedVarsData sharedVarsData, inout TResult res, in vec3 orig, in vec3 dir, in int tri, in bool isValid) {
    bool validTriangle = 
        isValid && 
        tri >= 0 && 
        tri != res.triangle &&
        tri != LONGEST;

    mat3 triverts = mat3(
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 0).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 1).xyz, 
        fetchMosaic(vertex_texture, gatherMosaic(getUniformCoord(tri)), 2).xyz
    );

    vec2 uv = vec2(0.0f);
    float _d = intersectTriangle(orig, dir, triverts, uv, validTriangle);
    bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.predist) && greaterEqualF(_d, 0.0f);
    bool inbaked = equalF(_d, 0.0f);
    bool isbaked = equalF(_d, res.predist);
    bool changed = !isbaked && !inbaked;

    if (near) {
        if ( changed ) {
            res.predist = _d;
            res.triangle = tri;
        }
        if ( inbaked ) {
            bakedStack[sharedVarsData.L][sharedVarsData.bakedStackCount++] = tri;
        } else 
        if ( sharedVarsData.bakedRange < tri || sharedVarsData.bakedRange == LONGEST || changed ) {
            sharedVarsData.bakedRange = tri;
            sharedVarsData.bakedRangeIntersection = vec4(_d, uv, 0.f);
        }
    }

    return res;
}

vec3 projectVoxels(in vec3 orig) {
     vec4 nps = mult4(vec4(orig, 1.0f), GEOMETRY_BLOCK octreeUniform.project);
    return nps.xyz / nps.w;
}

vec3 unprojectVoxels(in vec3 orig) {
     vec4 nps = mult4(vec4(orig, 1.0f), GEOMETRY_BLOCK octreeUniform.unproject);
    return nps.xyz / nps.w;
}

float intersectCubeSingle(in vec3 origin, in vec3 ray, in vec4 cubeMin, in vec4 cubeMax, inout float near, inout float far) {
     vec3 dr = 1.0f / ray;
     vec3 tMin = (cubeMin.xyz - origin) * dr;
     vec3 tMax = (cubeMax.xyz - origin) * dr;
     vec3 t1 = min(tMin, tMax);
     vec3 t2 = max(tMin, tMax);
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


void intersectCubeApart(in vec3 origin, in vec3 ray, in vec4 cubeMin, in vec4 cubeMax, inout float near, inout float far) {
     vec3 dr = 1.0f / ray;
     vec3 tMin = (cubeMin.xyz - origin) * dr;
     vec3 tMax = (cubeMax.xyz - origin) * dr;
     vec3 t1 = min(tMin, tMax);
     vec3 t2 = max(tMin, tMax);
#ifdef ENABLE_AMD_INSTRUCTION_SET
    near = max3(t1.x, t1.y, t1.z);
    far  = min3(t2.x, t2.y, t2.z);
#else
    near = max(max(t1.x, t1.y), t1.z);
    far  = min(min(t2.x, t2.y), t2.z);
#endif
}

const vec3 padding = vec3(0.00001f);
const int STACK_SIZE = 16;
shared int deferredStack[WORK_SIZE][STACK_SIZE];

bvec2 and2(in bvec2 a, in bvec2 b){
    //return a && b;
    return bvec2(a.x && b.x, a.y && b.y);
}

bvec2 or2(in bvec2 a, in bvec2 b){
    //return a || b;
    return bvec2(a.x || b.x, a.y || b.y);
}

bvec2 not2(in bvec2 a){
    //return !a;
    return bvec2(!a.x, !a.y);
}

TResult traverse(in float distn, in vec3 origin, in vec3 direct, in Hit hit) {
    const uint L = gl_LocalInvocationID.x;

    int idx = 0, deferredPtr = 0;
    bool validBox = false;
    bool skip = false;

    TResult lastRes;
    lastRes.dist = INFINITY;
    lastRes.predist = INFINITY;
    lastRes.triangle = LONGEST;
    lastRes.materialID = LONGEST;

    SharedVarsData sharedVarsData;
    sharedVarsData.bakedRange = LONGEST;
    sharedVarsData.bakedStackCount = 0;
    sharedVarsData.L = L;

    deferredStack[L][0] = -1;

    // test constants
     int bakedStep = int(floor(1.f + hit.vmods.w));
     vec3 torig = projectVoxels(origin);
     vec3 tdirproj = mult4(vec4(direct, 0.0), GEOMETRY_BLOCK octreeUniform.project).xyz;
     float dirlen = 1.0f / length(tdirproj);
     vec3 dirproj = normalize(tdirproj);

    // test with root node
    //HlbvhNode node = Nodes[idx];
    // bbox lbox = node.box;
    float near = INFINITY, far = INFINITY;
    // float d = intersectCubeSingle(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, near, far);
     float d = intersectCubeSingle(torig, dirproj, vec4(vec3(0.0f), 1.0f), vec4(1.0f), near, far);
    lastRes.predist = far * dirlen;

    // init state
    {
        validBox = lessF(d, INFINITY) && greaterEqualF(d, 0.0f);
    }

    for(int i=0;i<8192;i++) {
        //if (allInvocationsARB(!validBox)) break;
        if (!validBox) break;
        HlbvhNode node = Nodes[idx];

        // is leaf
         bool isLeaf = node.pdata.x == node.pdata.y && validBox;
        //if (anyInvocationARB(isLeaf)) {
        if (isLeaf) {
            testIntersection(sharedVarsData, lastRes, origin, direct, node.pdata.w, isLeaf);
        }

        bool notLeaf = node.pdata.x != node.pdata.y && validBox;
        //if (anyInvocationARB(notLeaf)) {
        if (notLeaf) {
            HlbvhNode lnode = Nodes[node.pdata.x];
            HlbvhNode rnode = Nodes[node.pdata.y];

            vec2 inf2 = vec2(INFINITY);
            vec2 nearsLR = inf2;
            vec2 farsLR = inf2;
            intersectCubeApart(torig, dirproj, lnode.box.mn, lnode.box.mx, nearsLR.x, farsLR.x);
            intersectCubeApart(torig, dirproj, rnode.box.mn, rnode.box.mx, nearsLR.y, farsLR.y);

             bvec2 isCube = and2(greaterThanEqual(farsLR, nearsLR), greaterThanEqual(farsLR, vec2(0.0f)));
             vec2 nears = mix(inf2, nearsLR, isCube);
             vec2  fars = mix(inf2, farsLR, isCube);
             vec2  hits = mix(nears, fars, lessThan(nears, vec2(0.0f)));

             bvec2 overlaps = 
                and2(bvec2(notLeaf), 
                and2(lessThanEqual(hits, vec2(INFINITY-PZERO)),
                and2(greaterThan(hits, -vec2(PZERO)),
                greaterThan(vec2(lastRes.predist), nears * dirlen - PZERO))));
            
             bool anyOverlap = any(overlaps);
            //if (anyInvocationARB(anyOverlap)) {
            if (anyOverlap) {
                 bool leftOrder = all(overlaps) ? lessEqualF(hits.x, hits.y) : overlaps.x;

                ivec2 leftright = mix(ivec2(-1), node.pdata.xy, overlaps);
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
            idx = valid ? exchange(deferredStack[L][ptr], -1) : -1;
            validBox = validBox && valid && idx >= 0;
        } skip = false;
    }

    choiceBaked(sharedVarsData, lastRes, origin, direct, bakedStep);
    return lastRes;
}