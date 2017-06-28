layout ( std430, binding = 9 ) readonly buffer NodesBlock { HlbvhNode Nodes[]; };

vec4 bakedRangeIntersection[1];
int bakedRange[1];

const int bakedFragments = 8;
int bakedStack[bakedFragments];
int bakedStackCount = 0;

// WARP optimized triangle intersection
float intersectTriangle(in VEC3 orig, in VEC3 dir, in vec3 ve, inout vec2 UV) {
    const vec2 e12 = ve.yz - ve.x;

    bool valid = !(length3(e12.x) < 0.00001f && length3(e12.y) < 0.00001f);
    if (ibs(valid)) return INFINITY;

    const VEC3 pvec = cross3(dir, e12.y);
    const float det = dot3(e12.x, pvec);

#ifndef CULLING
    if (abs(det) <= 0.0f) valid = false;
#else
    if (det <= 0.0f) valid = false;
#endif
    if (ibs(valid)) return INFINITY;

    const VEC3 tvec = orig - ve.x;
    const float u = dot3(tvec, pvec);
    const VEC3 qvec = cross3(tvec, e12.x);
    const float v = dot3(dir, qvec);
    const float ts = dot3(e12.y, qvec);
    const VEC3 uvt = (eql(0) ? u : (eql(1) ? v : ts)) / det;

    const float vd = x(uvt);
    const bool invalidU = any2(uvt < VEC2(0.f));
    const bool invalidV = any2((VEC2(uvt) + (eql(0) ? 0.f : vd)) > VEC2(1.f));

    if (invalidU || invalidV) valid = false;
    if (ibs(valid)) return INFINITY;

    UV.xy = compvec2(uvt);
    const float dst = z(uvt);
    return (lessF(dst, 0.0f) || !valid) ? INFINITY : dst;
}

TResult choiceFirstBaked(inout TResult res) {
    const int tri = exchange(bakedRange[0], LONGEST);

    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;
        
    if (ibs(validTriangle)) return res;

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

TResult choiceBaked(inout TResult res, in VEC3 orig, in VEC3 dir, in int tpi) {
    choiceFirstBaked(res);
    reorderTriangles();

    const int tri = tpi < exchange(bakedStackCount, 0) ? bakedStack[tpi] : LONGEST;

    const bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;

    vec3 triverts;
    for (int x=0;x<3;x++) {
        const VEC3 swizVertex = swiz(verts[tri * 3 + x].vertex);
        putv3(validTriangle ? swizVertex : VEC3(0.0f), triverts, x);
    }

    vec2 uv = vec2(0.0f);
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

    vec3 triverts;
    for (int x=0;x<3;x++) {
        const VEC3 swizVertex = swiz(verts[tri * 3 + x].vertex);
        putv3(validTriangle ? swizVertex : VEC3(0.0f), triverts, x);
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
            bakedRangeIntersection[0] = vec4(_d, uv.xy, 0.0f);
        }
    }

    return res;
}

VEC3 projectVoxels(in VEC3 orig) {
    const VEC4 nps = mult4w(eql(3) ? 1.0f : orig, GEOMETRY_BLOCK octreeUniform.project);
    return nps / w(nps);
}

VEC3 unprojectVoxels(in VEC3 orig) {
    const VEC4 nps = mult4w(eql(3) ? 1.0f : orig, GEOMETRY_BLOCK octreeUniform.unproject);
    return nps / w(nps);
}

float intersectCubeSingle(in VEC3 origin, in VEC3 ray, in VEC3 cubeMin, in VEC3 cubeMax, inout float near, inout float far) {
    const VEC3 dr = 1.0f / ray;
    const VEC3 tMin = (cubeMin - origin) * dr;
    const VEC3 tMax = (cubeMax - origin) * dr;
    const VEC3 t1 = min(tMin, tMax);
    const VEC3 t2 = max(tMin, tMax);
    const float tNear = compmax3(t1);
    const float tFar  = compmin3(t2);
    const bool isCube = tFar >= tNear && greaterEqualF(tFar, 0.0f);
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}




// optimized by WARP and SIMD intersection boxes

vec2 intersectCubeSingleApart(in VEC3 origin, in VEC3 ray, in VEC3 cubeMin, in VEC3 cubeMax) {
    const VEC3 dr = 1.0f / ray;
    const VEC3 tMin = (cubeMin - origin) * dr;
    const VEC3 tMax = (cubeMax - origin) * dr;
    const VEC3 t1 = min(tMin, tMax);
    const VEC3 t2 = max(tMin, tMax);
    return vec2(t1, t2);
}

float calculateRealDistance(in float tNear, in float tFar, inout float near, inout float far) {
    const bool isCube = tFar >= tNear && greaterEqualF(tFar, 0.0f);
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}



const VEC3 padding = VEC3(0.00001f);
const int STACK_SIZE = 32;
shared int deferredStack[WORK_SIZE][STACK_SIZE];
shared int deferredPtr[WORK_SIZE];
shared int idx[WORK_SIZE];
shared bool skip[WORK_SIZE];
shared bool validBox[WORK_SIZE];

//TResult traverse(in float distn, in VEC3 origin, in VEC3 direct, in Hit hit) {
TResult traverse(in float distn, in vec3 _origin, in vec3 _direct, in Hit hit) {
    const uint L = invoc(gl_LocalInvocationID.x);
    const VEC4 origin = eql(3) ? 1.0f : swiz(_origin); 
    const VEC4 direct = eql(3) ? 0.0f : swiz(_direct);

    TResult lastRes;
    lastRes.dist = INFINITY;
    lastRes.predist = INFINITY;
    lastRes.triangle = LONGEST;
    lastRes.materialID = LONGEST;
    bakedRange[0] = LONGEST;

    // test constants
    const int bakedStep = int(floor(1.f + hit.vmods.w));
    const VEC4 torig = projectVoxels(origin);
    const VEC4 tdirproj = mult4w(direct, GEOMETRY_BLOCK octreeUniform.project);
    const float dirlen = 1.0f / length3(tdirproj);
    const VEC4 dirproj = normalize3(tdirproj);
    
    const vec3 torigUnif = compvec3(torig);
    const vec3 dirprojUnif = compvec3(dirproj);

    // test with root node
    //HlbvhNode node = Nodes[idx];
    //const bbox lbox = node.box;
    float near = INFINITY, far = INFINITY;
    //const float d = intersectCubeSingle(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, near, far);
    const float d = intersectCubeSingle(torig, dirproj, VEC3(0.0f), VEC3(1.0f), near, far);
    lastRes.predist = far * dirlen;

    // init state
    if (mt()) {
        idx[L] = 0, deferredPtr[L] = 0;
        validBox[L] = 
            lessF(d, INFINITY) 
            && greaterEqualF(d, 0.0f) 
            && greaterEqualF(lastRes.predist, near * dirlen)
            ;
    }

    for(int i=0;i<16384;i++) {
        if (ibs(validBox[L])) break;
        HlbvhNode node = Nodes[idx[L]];
        testIntersection(lastRes, origin, direct, node.triangle, bakedStep, node.range.x == node.range.y && validBox[L]);

        bool notLeaf = node.range.x != node.range.y && validBox[L];
        if (bs(notLeaf)) {

            // search intersection worklets (3 lanes occupy)
            const vec2 t12_leftright[2] = {
                intersectCubeSingleApart(torig, dirproj, swiz(Nodes[node.range.x].box.mn), swiz(Nodes[node.range.x].box.mx)),
                intersectCubeSingleApart(torig, dirproj, swiz(Nodes[node.range.y].box.mn), swiz(Nodes[node.range.y].box.mx))
            };
            
            // transpose from both nodes
            const float hnears[2] = {
                compmax3(t12_leftright[0].x),
                compmax3(t12_leftright[1].x)
            };

            const float hfars[2] = {
                compmin3(t12_leftright[0].y),
                compmin3(t12_leftright[1].y)
            };

            // determine parallel (2 lanes occupy)
            float nearLR = INFINITY, farLR = INFINITY;
            const int distrb = eql(0) ? 0 : 1;
            const float leftrighthit = calculateRealDistance(hnears[distrb], hfars[distrb], nearLR, farLR);
            const bool overlapsVc = notLeaf && lessF(leftrighthit, INFINITY) && greaterEqualF(leftrighthit, 0.0f) && greaterEqualF(lastRes.predist, near * dirlen);
            
            // compose results
            const bvec2 overlaps = compbvec2(overlapsVc);
            const bool overlapAny = any(overlaps);
            if (bs(overlapAny)) {
                const bool leftNearb = lessEqualF(x(leftrighthit), y(leftrighthit));
                if (overlapAny && mt()) {
                    const bool leftOrder = all(overlaps) ? leftNearb : overlaps.x;
                    ivec2 leftright = mix(ivec2(-1), node.range.xy, overlaps);
                    leftright = leftOrder ? leftright : leftright.yx;
                    if (deferredPtr[L] < STACK_SIZE && leftright.y != -1) {
                        deferredStack[L][deferredPtr[L]++] = leftright.y;
                    }
                    idx[L] = leftright.x;
                    skip[L] = true;
                }
            }
        }

        if (mt()) {
            if (!skip[L]) {
                const int ptr = --deferredPtr[L];
                const bool valid = ptr >= 0;
                {
                    idx[L] = valid ? exchange(deferredStack[L][ptr], -1) : -1;
                }
                validBox[L] = validBox[L] && valid && idx[L] >= 0;
            } skip[L] = false;
        }
    }

    choiceBaked(lastRes, origin, direct, bakedStep);
    return loadInfo(lastRes);
}
