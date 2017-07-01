layout ( std430, binding = 9 ) readonly buffer NodesBlock { HlbvhNode Nodes[]; };

vec4 bakedRangeIntersection[1];
int bakedRange[1];

const int bakedFragments = 8;
int bakedStack[bakedFragments];
int bakedStackCount = 0;

// WARP optimized triangle intersection
float intersectTriangle(in vec3 orig, in vec3 dir, in vec3 ve[3], inout vec2 UV) {
    const vec3 e1 = ve[1] - ve[0];
    const vec3 e2 = ve[2] - ve[0];

    bool valid = !(length(e1) < 0.00001f && length(e2) < 0.00001f);
    if (allInvocationsARB(!valid)) return INFINITY;

    const vec3 pvec = cross(dir, e2);
    const float det = dot(e1, pvec);

#ifndef CULLING
    if (abs(det) <= 0.0f) valid = false;
#else
    if (det <= 0.0f) valid = false;
#endif
    if (allInvocationsARB(!valid)) return INFINITY;

    const vec3 tvec = orig - ve[0];
    const float u = dot(tvec, pvec);
    const vec3 qvec = cross(tvec, e1);
    const float v = dot(dir, qvec);
    const vec3 uvt = vec3(u, v, dot(e2, qvec)) / det;

    if (
        any(lessThan(uvt.xy, vec2(0.f))) || 
        any(greaterThan(vec2(uvt.x) + vec2(0.f, uvt.y), vec2(1.f))) 
    ) valid = false;
    if (allInvocationsARB(!valid)) return INFINITY;

    UV.xy = uvt.xy;
    return (lessF(uvt.z, 0.0f) || !valid) ? INFINITY : uvt.z;
}

TResult choiceFirstBaked(inout TResult res) {
    const int tri = exchange(bakedRange[0], LONGEST);

    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;
        
    //if (!validTriangle) return res;
    if (allInvocationsARB(!validTriangle)) return res;

    const vec2 uv = bakedRangeIntersection[0].yz;
    const float _d = bakedRangeIntersection[0].x;
    const bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv.xy;
    }

    return res;
}

void reorderTriangles() {
    bakedStackCount = min(bakedStackCount, bakedFragments);
    for (int round = 1; round < bakedStackCount; round++) {
        for (int index = 0; index < bakedStackCount - round; index++) {
            if (bakedStack[index] <= bakedStack[index+1]) {
                swap(bakedStack[index], bakedStack[index+1]);
            }
        }
    }

    // clean from dublicates
    int cleanBakedStack[bakedFragments];
    int cleanBakedStackCount = 0;
    cleanBakedStack[cleanBakedStackCount++] = bakedStack[0];
    for (int round = 1; round < bakedStackCount; round++) {
        if(bakedStack[round-1] != bakedStack[round]) {
            cleanBakedStack[cleanBakedStackCount++] = bakedStack[round];
        }
    }

    // set originally
    bakedStack = cleanBakedStack;
    bakedStackCount = cleanBakedStackCount;
}

TResult choiceBaked(inout TResult res, in vec3 orig, in vec3 dir, in int tpi) {
    choiceFirstBaked(res);
    reorderTriangles();

    const int tri = tpi < exchange(bakedStackCount, 0) ? bakedStack[tpi] : LONGEST;

    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;

    vec3 triverts[3];
    for (int x=0;x<3;x++) {
        triverts[x] = validTriangle ? vec3(verts[tri * 3 + x].vertex) : vec3(0.0f);
    }

    vec2 uv = vec2(0.0f);
    const float _d = intersectTriangle(orig, dir, triverts, uv);
    const bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv.xy;
    }
    
    return res;
}

TResult testIntersection(inout TResult res, in vec3 orig, in vec3 dir, in int tri, in bool isValid) {
    const bool validTriangle = 
        isValid && 
        tri >= 0 && 
        tri != res.triangle &&
        tri != LONGEST;

    vec3 triverts[3];
    for (int x=0;x<3;x++) {
        triverts[x] = validTriangle ? vec3(verts[tri * 3 + x].vertex) : vec3(0.0f);
    }

    vec2 uv = vec2(0.0f);
    const float _d = intersectTriangle(orig, dir, triverts, uv);
    const bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.predist) && greaterEqualF(_d, 0.0f);
    const bool inbaked = equalF(_d, 0.0f);
    const bool isbaked = equalF(_d, res.predist);
    const bool changed = !isbaked && !inbaked;

    if (near) {
        if ( changed ) {
            res.predist = _d;
            res.triangle = tri;
        }
        if ( inbaked ) {
            bakedStack[bakedStackCount++] = tri;
        } else 
        if ( bakedRange[0] < tri || bakedRange[0] == LONGEST || changed ) {
            bakedRange[0] = tri;
            bakedRangeIntersection[0] = vec4(_d, uv, 0.f);
        }
    }

    return res;
}

vec3 projectVoxels(in vec3 orig) {
    const vec4 nps = mult4(vec4(orig, 1.0f), GEOMETRY_BLOCK octreeUniform.project);
    return nps.xyz / nps.w;
}

vec3 unprojectVoxels(in vec3 orig) {
    const vec4 nps = mult4(vec4(orig, 1.0f), GEOMETRY_BLOCK octreeUniform.unproject);
    return nps.xyz / nps.w;
}

float intersectCubeSingle(in vec3 origin, in vec3 ray, in vec3 cubeMin, in vec3 cubeMax, inout float near, inout float far) {
    const vec3 dr = 1.0f / ray;
    const vec3 tMin = (cubeMin - origin) * dr;
    const vec3 tMax = (cubeMax - origin) * dr;
    const vec3 t1 = min(tMin, tMax);
    const vec3 t2 = max(tMin, tMax);
#ifdef ENABLE_AMD_INSTRUCTION_SET
    const float tNear = max3(t1.x, t1.y, t1.z);
    const float tFar  = min3(t2.x, t2.y, t2.z);
#else
    const float tNear = max(max(t1.x, t1.y), t1.z);
    const float tFar  = min(min(t2.x, t2.y), t2.z);
#endif
    const bool isCube = tFar >= tNear && greaterEqualF(tFar, 0.0f);
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}


void intersectCubeApart(in vec3 origin, in vec3 ray, in vec3 cubeMin, in vec3 cubeMax, inout float near, inout float far) {
    const vec3 dr = 1.0f / ray;
    const vec3 tMin = (cubeMin - origin) * dr;
    const vec3 tMax = (cubeMax - origin) * dr;
    const vec3 t1 = min(tMin, tMax);
    const vec3 t2 = max(tMin, tMax);
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
int deferredStack[STACK_SIZE];
int idx = 0, deferredPtr = 0;
bool validBox = false;

TResult traverse(in float distn, in vec3 origin, in vec3 direct, in Hit hit) {
    TResult lastRes;
    lastRes.dist = INFINITY;
    lastRes.predist = INFINITY;
    lastRes.triangle = LONGEST;
    lastRes.materialID = LONGEST;
    bakedRange[0] = LONGEST;

    // test constants
    const int bakedStep = int(floor(1.f + hit.vmods.w));
    const vec3 torig = projectVoxels(origin);
    const vec3 tdirproj = mult4(vec4(direct, 0.0), GEOMETRY_BLOCK octreeUniform.project).xyz;
    const float dirlen = 1.0f / length(tdirproj);
    const vec3 dirproj = normalize(tdirproj);

    // test with root node
    //HlbvhNode node = Nodes[idx];
    //const bbox lbox = node.box;
    float near = INFINITY, far = INFINITY;
    //const float d = intersectCubeSingle(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, near, far);
    const float d = intersectCubeSingle(torig, dirproj, vec3(0.0f), vec3(1.0f), near, far);
    lastRes.predist = far * dirlen;

    // init state
    {
        idx = 0, deferredPtr = 0;
        validBox = 
            lessF(d, INFINITY) 
            && greaterEqualF(d, 0.0f) 
            && greaterEqualF(lastRes.predist, near * dirlen)
            ;
    }

    bool skip = false;
    for(int i=0;i<16384;i++) {
        if (allInvocationsARB(!validBox)) break;
        const HlbvhNode node = Nodes[idx];

        // is leaf
        const bool isLeaf = node.range.x == node.range.y && validBox;
        if (anyInvocationARB(isLeaf)) {
            testIntersection(lastRes, origin, direct, node.triangle, isLeaf);
        }

        bool notLeaf = node.range.x != node.range.y && validBox;
        if (anyInvocationARB(notLeaf)) {
            const bbox lbox = Nodes[node.range.x].box;
            const bbox rbox = Nodes[node.range.y].box;
            const vec2 inf2 = vec2(INFINITY);
            vec2 nearsLR = inf2;
            vec2 farsLR = inf2;
            intersectCubeApart(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, nearsLR.x, farsLR.x);
            intersectCubeApart(torig, dirproj, rbox.mn.xyz, rbox.mx.xyz, nearsLR.y, farsLR.y);

            const bvec2 isCube = greaterThanEqual(farsLR, nearsLR) && greaterThanEqual(farsLR, vec2(0.0f));
            const vec2 nears = mix(inf2, nearsLR, isCube);
            const vec2  fars = mix(inf2, farsLR, isCube);
            const vec2  hits = mix(nears, fars, lessThan(nears, vec2(0.0f)));

            const bvec2 overlaps = 
                bvec2(notLeaf) && 
                lessThanEqual(hits, vec2(INFINITY-PZERO)) && 
                greaterThan(hits, -vec2(PZERO)) && 
                greaterThan(vec2(lastRes.predist), nears * dirlen - PZERO);
            
            const bool anyOverlap = any(overlaps);
            if (anyInvocationARB(anyOverlap)) {
                const bool leftOrder = all(overlaps) ? lessEqualF(hits.x, hits.y) : overlaps.x;

                ivec2 leftright = mix(ivec2(-1), node.range, overlaps);
                leftright = leftOrder ? leftright : leftright.yx;

                if (anyOverlap) {
                    if (deferredPtr < STACK_SIZE && leftright.y != -1) {
                        deferredStack[deferredPtr++] = leftright.y;
                    }
                    idx = leftright.x;
                    skip = true;
                }
            }
        }

        if (!skip) {
            const int ptr = --deferredPtr;
            const bool valid = ptr >= 0;
            idx = valid ? exchange(deferredStack[ptr], -1) : -1;
            validBox = validBox && valid && idx >= 0;
        } skip = false;
    }

    choiceBaked(lastRes, origin, direct, bakedStep);
    return loadInfo(lastRes);
}
