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
float intersectTriangle(in VEC3 orig, in VEC3 dir, in VEC3 ve[3], inout vec2 UV) {
    const VEC3 e1 = ve[1] - ve[0];
    const VEC3 e2 = ve[2] - ve[0];

    bool valid = !(
           length3(e1) < 0.00001f 
        && length3(e2) < 0.00001f
    );
    if (ballotARB(valid) == 0) return INFINITY;

    const VEC3 pvec = cross3(dir, e2);
    const float det = dot3(e1, pvec);

#ifndef CULLING
    if (abs(det) <= 0.0f) valid = false;
#else
    if (det <= 0.0f) valid = false;
#endif
    if (ballotARB(valid) == 0) return INFINITY;

    const VEC3 tvec = orig - ve[0];
    const float u = dot3(tvec, pvec);
    const VEC3 qvec = cross3(tvec, e1);
    const float v = dot3(dir, qvec);
    const float ts = dot3(e2, qvec);
    const VEC3 uvt = (eql(0) ? u : (eql(1) ? v : ts)) / det;

    const float vd = x(uvt);
    const bool invalidU = any2(uvt < VEC2(0.f));
    const bool invalidV = any2((VEC2(uvt) + (eql(0) ? 0.f : vd)) > VEC2(1.f));

    if (invalidU || invalidV) valid = false;
    if (ballotARB(valid) == 0) return INFINITY;

    UV.xy = compvec2(uvt);
    const float dst = z(uvt);
    return (lessF(dst, 0.0f) || !valid) ? INFINITY : dst;
}

float intersectTriangle(in VEC3 orig, in VEC3 dir, inout VEC3 ve[3]) {
    vec2 uv = vec2(0.0f);
    return intersectTriangle(orig, dir, ve, uv);
}

TResult choiceFirstBaked(inout TResult res, in VEC3 orig, in VEC3 dir) {
    const int tri = bakedRange[0];
    bakedRange[0] = LONGEST;

    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;
        
    //if (!validTriangle) return res;
    if (ballotARB(validTriangle) == 0) return res;

    const vec2 uv = bakedRangeIntersection[0].yz;
    const float _d = bakedRangeIntersection[0].x;
    const bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv;
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
}

TResult choiceBaked(inout TResult res, in VEC3 orig, in VEC3 dir, in int tpi) {
    choiceFirstBaked(res, orig, dir);
    reorderTriangles();

    const int tri = tpi < bakedStackCount ? bakedStack[tpi] : LONGEST;
    bakedStackCount = 0;

    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;

    vec2 uv = vec2(0.0f);
    VEC3 triverts[3];

#pragma optionNV (unroll all)
    for (int x=0;x<3;x++) {
        triverts[x] = validTriangle ? swiz(verts[tri * 3 + x].vertex) : VEC3(0.0f);
    }

    const float _d = intersectTriangle(orig, dir, triverts, uv);
    const bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv;
    }
    
    return res;
}

TResult testIntersection(inout TResult res, in VEC3 orig, in VEC3 dir, in int tri, in int step, in bool isValid) {
    const bool validTriangle = 
        isValid && 
        tri >= 0 && 
        tri != LONGEST;

    vec2 uv = vec2(0.0f);
    VEC3 triverts[3];

#pragma optionNV (unroll all)
    for (int x=0;x<3;x++) {
        triverts[x] = validTriangle ? swiz(verts[tri * 3 + x].vertex) : VEC3(0.0f);
    }

    const float _d = intersectTriangle(orig, dir, triverts, uv);
    const bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.predist) && greaterEqualF(_d, 0.0f);
    const bool inbaked = equalF(_d, 0.0f);
    const bool isbaked = equalF(_d, res.predist);
    const bool changed = !isbaked && !inbaked;

    if (near) {
        if (changed) res.predist = _d;

        if ( inbaked ) {
            bakedStack[bakedStackCount++] = tri;
        } else 
        if ( bakedRange[0] < tri || bakedRange[0] == LONGEST || changed ) {
            bakedRange[0] = tri;
            bakedRangeIntersection[0] = vec4(_d, uv.xy, 0.0f);
        }
        
    }

    return res;
}

VEC3 projectVoxels(in VEC3 orig) {
    const VEC4 nps = mult4w(GEOMETRY_BLOCK octreeUniform.project, eql(3) ? 1.0f : orig);
    return nps / w(nps);
}

VEC3 unprojectVoxels(in VEC3 orig) {
    const VEC4 nps = mult4w(GEOMETRY_BLOCK octreeUniform.unproject, eql(3) ? 1.0f : orig);
    return nps / w(nps);
}

float intersectCubeSingle(in VEC3 origin, in VEC3 ray, in VEC3 cubeMin, in VEC3 cubeMax, inout float near, inout float far) {
    const VEC3 dr = 1.0f / ray;
    const VEC3 tMin = (cubeMin - origin) * dr;
    const VEC3 tMax = (cubeMax - origin) * dr;
    const VEC3 t1 = min(tMin, tMax);
    const VEC3 t2 = max(tMin, tMax);
    const float tNear = max(max(x(t1), y(t1)), z(t1));
    const float tFar  = min(min(x(t2), y(t2)), z(t2));
    const bool isCube = tFar >= tNear && greaterEqualF(tFar, 0.0f);
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}

const VEC3 padding = VEC3(0.00001f);
const int STACK_SIZE = 32;
int deferredStack[STACK_SIZE];

//TResult traverse(in float distn, in VEC3 origin, in VEC3 direct, in Hit hit) {
TResult traverse(in float distn, in vec3 _origin, in vec3 _direct, in Hit hit) {
    VEC3 origin = swiz(_origin); VEC3 direct = swiz(_direct);
    if (eql(3)) {
        origin = 1.0f;
        direct = 0.0f;
    }

    TResult lastRes;
    lastRes.dist = INFINITY;
    lastRes.predist = INFINITY;
    lastRes.triangle = LONGEST;
    lastRes.materialID = LONGEST;
    bakedRange[0] = LONGEST;

    // test constants
    const int bakedStep = int(floor(1.f + hit.vmods.w));
    const VEC3 torig = projectVoxels(origin);
    const VEC3 tdirproj = mult4w(GEOMETRY_BLOCK octreeUniform.project, direct);
    const float dirlen = 1.0f / length3(tdirproj);
    const VEC3 dirproj = normalize3(tdirproj);

    // init state
    int idx = 0, deferredPtr = 0;

    // test with root node
    //HlbvhNode node = Nodes[idx];
    //const bbox lbox = node.box;
    float near = INFINITY, far = INFINITY;
    //const float d = intersectCubeSingle(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, near, far);
    const float d = intersectCubeSingle(torig, dirproj, VEC3(0.0f), VEC3(1.0f), near, far);
    lastRes.predist = far * dirlen;

    bool validBox = 
        lessF(d, INFINITY) 
        && greaterEqualF(d, 0.0f) 
        && greaterEqualF(lastRes.predist, near * dirlen)
        ;

    bool skip = false;
    for(int i=0;i<16384;i++) {
        if (ballotARB(validBox) == 0) break;
        HlbvhNode node = Nodes[idx];
        testIntersection(lastRes, origin, direct, node.triangle, bakedStep, node.range.x == node.range.y && validBox);

        bool notLeaf = node.range.x != node.range.y && validBox;
        //if (node.range.x != node.range.y && validBox) {
        if (ballotARB(notLeaf) > 0) {
            bool leftOverlap = false, rightOverlap = false;
            float lefthit = INFINITY, righthit = INFINITY;

            {
                float near = INFINITY, far = INFINITY;
                const bbox lbox = Nodes[node.range.x].box;
                lefthit = intersectCubeSingle(torig, dirproj, swiz(lbox.mn.xyz), swiz(lbox.mx.xyz), near, far);
                leftOverlap = 
                    notLeaf 
                    && lessF(lefthit, INFINITY) 
                    && greaterEqualF(lefthit, 0.0f) 
                    && greaterEqualF(lastRes.predist, near * dirlen)
                    ;
            }

            {
                float near = INFINITY, far = INFINITY;
                const bbox rbox = Nodes[node.range.y].box;
                righthit = intersectCubeSingle(torig, dirproj, swiz(rbox.mn.xyz), swiz(rbox.mx.xyz), near, far);
                rightOverlap = 
                    notLeaf 
                    && lessF(righthit, INFINITY) 
                    && greaterEqualF(righthit, 0.0f) 
                    && greaterEqualF(lastRes.predist, near * dirlen)
                    ;
            }

            const BVEC2 overlaps = eql(0) ? leftOverlap : rightOverlap;
            const bool overlapAny = leftOverlap || rightOverlap;//any2(overlaps);
            const bool overlapAll = leftOverlap && rightOverlap;//all2(overlaps);
            const bool leftOvp = leftOverlap;//x(overlaps);

            if (ballotARB(overlapAny) > 0) {
                IVEC2 leftright = mix(IVEC2(-1), swiz(node.range.xy), overlaps);

                // order by distance or valid
                const bool leftOrder = overlapAll ? lessEqualF(lefthit, righthit) : leftOvp;
                const IVEC2 swappedLR = swapXY(leftright);
                leftright = leftOrder ? leftright : swappedLR;

                const int nearer = x(leftright);
                const int ranger = y(leftright);
                if (overlapAny) {
                    idx = nearer;
                    if (deferredPtr < STACK_SIZE && ranger != -1) {
                        deferredStack[deferredPtr++] = ranger;
                    }
                    skip = true;
                }
            }
        }

        if (!skip) {
            const int ptr = --deferredPtr;
            const bool valid = ptr >= 0;

            idx = valid ? deferredStack[ptr] : -1;
            if (valid) deferredStack[ptr] = -1;

            validBox = validBox && valid && idx >= 0;
        } skip = false;
    }

    choiceBaked(lastRes, origin, direct, bakedStep);
    return loadInfo(lastRes);
}
