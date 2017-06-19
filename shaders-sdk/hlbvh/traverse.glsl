layout ( std430, binding = 9 ) readonly buffer NodesBlock { HlbvhNode Nodes[]; };

vec4 bakedRangeIntersection[1];
int bakedRange[1];

const int bakedFragments = 16;
int bakedStack[bakedFragments];
int bakedStackCount = 0;

void swap(inout int a, inout int b){
    const int t = a;
    a = b;
    b = t;
}

// WARP optimized triangle intersection
float intersectTriangle(in vec3 orig, in vec3 dir, inout vec3 ve[3], inout vec2 UV) {
    const vec3 e1 = ve[1] - ve[0];
    const vec3 e2 = ve[2] - ve[0];

    if (
        lessEqualF(length(e1), 0.f) && 
        lessEqualF(length(e2), 0.f)
    ) return INFINITY;

    const vec3 pvec = cross(dir, e2);
    const float det = dot(e1, pvec);

#ifndef CULLING
    if (abs(det) < 0.00001f) return INFINITY;
#else
    if (det < 0.00001f) return INFINITY;
#endif

    const vec3 tvec = orig - ve[0];
    const float u = dot(tvec, pvec);
    const vec3 qvec = cross(tvec, e1);
    const float v = dot(dir, qvec);
    const vec3 uvt = vec3(u, v, dot(e2, qvec)) / det;

    if (
        any(lessThan(uvt.xy, vec2(0.f))) || 
        any(greaterThan(vec2(uvt.x) + vec2(0.f, uvt.y), vec2(1.f))) 
    ) return INFINITY;

    UV.xy = uvt.xy;
    return (lessF(uvt.z, 0.0f)) ? INFINITY : uvt.z;
}

float intersectTriangle(in vec3 orig, in vec3 dir, inout vec3 ve[3]) {
    vec2 uv = vec2(0.0f);
    return intersectTriangle(orig, dir, ve, uv);
}

TResult choiceFirstBaked(inout TResult res, in vec3 orig, in vec3 dir) {
    const int tri = bakedRange[0];
    bakedRange[0] = LONGEST;

    const bool validTriangle = tri != LONGEST;
    if (!validTriangle) return res;

    const vec2 uv = bakedRangeIntersection[0].yz;
    const float _d = bakedRangeIntersection[0].x;
    const bool near = 
        validTriangle && 
        lessF(_d, INFINITY) &&
        lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv.xy;
    }

    return res;
}

void reorderTriangles() {
    bakedStackCount = min(bakedStackCount, bakedFragments);
    if (bakedStackCount <= 1) return;
    for (int round = 1; round < bakedStackCount; round++) {
        for (int index = 0; index < bakedStackCount - round; index++) {
            if (bakedStack[index] <= bakedStack[index+1]) {
                swap(bakedStack[index], bakedStack[index+1]);
            }
        }
    }
}

TResult choiceBaked(inout TResult res, in vec3 orig, in vec3 dir, in int tpi) {
    choiceFirstBaked(res, orig, dir);
    reorderTriangles();

    const int tri = tpi < bakedStackCount ? bakedStack[tpi] : LONGEST;
    bakedStackCount = 0;
    bakedStack[0] = LONGEST;

    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST && 
        tri < GEOMETRY_BLOCK geometryUniform.triangleCount;

    vec2 uv = vec2(0.0f);
    vec3 triverts[3];

#pragma optionNV (unroll all)
    for (int x=0;x<3;x++) {
        triverts[x] = validTriangle ? toVec3(verts[tri * 3 + x].vertex) : vec3(0.0f);
    }

    const float _d = intersectTriangle(orig, dir, triverts, uv);
    const bool near = lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv.xy;
    }
    
    return res;
}

TResult testIntersection(inout TResult res, in vec3 orig, in vec3 dir, in int tri, in int step) {
    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST && 
        tri < GEOMETRY_BLOCK geometryUniform.triangleCount;

    vec2 uv = vec2(0.0f);
    vec3 triverts[3];

#pragma optionNV (unroll all)
    for (int x=0;x<3;x++) {
        triverts[x] = validTriangle ? toVec3(verts[tri * 3 + x].vertex) : vec3(0.0f);
    }

    const float _d = intersectTriangle(orig, dir, triverts, uv);
    const bool near = lessF(_d, INFINITY) && lessEqualF(_d, res.predist);
    const bool inbaked = equalF(_d, 0.0f);
    const bool isbaked = !inbaked || step < bakedStackCount;
    const bool changed = !equalF(_d, res.predist) && isbaked;

    if (near) {
        if (changed) {
            res.predist = _d;
            bakedRange[0] = LONGEST;
        }
        if (inbaked) {
            bakedStack[bakedStackCount++] = tri;
        } else 
        if (tri > bakedRange[0] || bakedRange[0] == LONGEST) {
            bakedRange[0] = tri;
            bakedRangeIntersection[0] = vec4(_d, uv, 0.f);
        }
    }

    return res;
}

vec3 projectVoxels(in vec3 orig) {
    const vec4 nps = GEOMETRY_BLOCK octreeUniform.project * vec4(orig, 1.0f);
    return nps.xyz / nps.w;
}

vec3 unprojectVoxels(in vec3 orig) {
    const vec4 nps = GEOMETRY_BLOCK octreeUniform.unproject * vec4(orig, 1.0f);
    return nps.xyz / nps.w;
}

float intersectCubeSingle(in vec3 origin, in vec3 ray, in vec3 cubeMin, in vec3 cubeMax, inout float near, inout float far) {
    const vec3 dr = 1.0f / ray;
    const vec3 tMin = (cubeMin - origin) * dr;
    const vec3 tMax = (cubeMax - origin) * dr;
    const vec3 t1 = min(tMin, tMax);
    const vec3 t2 = max(tMin, tMax);
    const float tNear = max(max(t1.x, t1.y), t1.z);
    const float tFar  = min(min(t2.x, t2.y), t2.z);
    const bool isCube = tFar >= tNear && tFar >= 0.0f;
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessEqualF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}

const int STACK_SIZE = 32;
int deferredStack[STACK_SIZE];
int deferredPtr = 0;

TResult traverse(in float distn, in vec3 origin, in vec3 direct, in Hit hit) {
    TResult lastRes;
    lastRes.dist = INFINITY;
    lastRes.predist = INFINITY;
    lastRes.triangle = LONGEST;
    lastRes.materialID = LONGEST;
    bakedRange[0] = -1;
    deferredPtr = 0;

    direct = normalize(direct);
    vec3 dirproj = (GEOMETRY_BLOCK octreeUniform.project * vec4(direct, 0.0)).xyz;
    const float dirlen = 1.0f / length(dirproj);
    dirproj = normalize(dirproj);

    int idx = 0;
    HlbvhNode node = Nodes[idx];
    const bbox lbox = node.box; // need to use this fox, because with [0-1] box shows holes

    float near = INFINITY, far = INFINITY;
    const vec3 torig = projectVoxels(origin);
    const float d = intersectCubeSingle(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, near, far);
    const int bakedStep = int(floor(1.f + hit.vmods.w));
    lastRes.predist = far * dirlen;

    bool validBox = lessF(d, INFINITY) && greaterEqualF(d, 0.0f) && greaterEqualF(lastRes.predist, near * dirlen);
    if (!validBox) {
        return loadInfo(lastRes);
    }
    
    for(int i=0;i<16384;i++) {
        HlbvhNode node = Nodes[idx];

        if (node.range.x == node.range.y && validBox) {
            testIntersection(lastRes, origin, direct, node.triangle, bakedStep);
        }

        if (node.range.x != node.range.y && validBox) {
            bool leftOverlap = false, rightOverlap = false;
            float lefthit = INFINITY, righthit = INFINITY;

            {
                float near = INFINITY, far = INFINITY;
                const bbox lbox = Nodes[node.range.x].box;
                lefthit = intersectCubeSingle(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, near, far);
                leftOverlap = 
                    lessF(lefthit, INFINITY) &&
                    greaterEqualF(lefthit, 0.0f) &&
                    greaterEqualF(lastRes.predist, near * dirlen);
            }

            {
                float near = INFINITY, far = INFINITY;
                const bbox rbox = Nodes[node.range.y].box;
                righthit = intersectCubeSingle(torig, dirproj, rbox.mn.xyz, rbox.mx.xyz, near, far);
                rightOverlap = 
                    lessF(righthit, INFINITY) &&
                    greaterEqualF(righthit, 0.0f) &&
                    greaterEqualF(lastRes.predist, near * dirlen);
            }

            if (leftOverlap || rightOverlap) {
                const int leftNeeded  = leftOverlap  ? node.range.x : -1;
                const int rightNeeded = rightOverlap ? node.range.y : -1;

                if (leftOverlap && rightOverlap) {
                    leftOverlap = lefthit <= righthit;
                }
                
                const int farest  = leftOverlap ? rightNeeded : leftNeeded;
                const int nearest = leftOverlap ? leftNeeded : rightNeeded;

                idx = nearest;
                if (deferredPtr < STACK_SIZE && farest != -1) {
                    deferredStack[deferredPtr++] = farest;
                }
                continue;
            }
        }

        idx = deferredStack[--deferredPtr];

        const bool overhead = idx < 0 || deferredPtr < 0;
        validBox = validBox && !overhead;
        if ( overhead ) { break; }
    }

    choiceBaked(lastRes, origin, direct, bakedStep);
    return loadInfo(lastRes);
}