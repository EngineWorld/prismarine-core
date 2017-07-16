layout ( std430, binding = 9 ) readonly buffer NodesBlock { HlbvhNode Nodes[]; };

vec4 bakedRangeIntersection[1];
int bakedRange[1];

const int bakedFragments = 8;
int bakedStack[bakedFragments];
int bakedStackCount = 0;

// WARP optimized triangle intersection
float intersectTriangle(in VEC3 orig, in VEC3 dir, in vec3 ve, inout vec2 UV) {
     vec2 e12 = ve.yz - ve.x;

    bool valid = !(length3(e12.x) < 0.00001f && length3(e12.y) < 0.00001f);
    if (ibs(valid)) return INFINITY;

     VEC3 pvec = cross3(dir, e12.y);
     float det = dot3(e12.x, pvec);

#ifndef CULLING
    if (abs(det) <= 0.0f) valid = false;
#else
    if (det <= 0.0f) valid = false;
#endif
    if (ibs(valid)) return INFINITY;

     VEC3 tvec = orig - ve.x;
     float u = dot3(tvec, pvec);
     VEC3 qvec = cross3(tvec, e12.x);
     float v = dot3(dir, qvec);
     float ts = dot3(e12.y, qvec);
     VEC3 uvt = (eql(0) ? u : (eql(1) ? v : ts)) / det;

     float vd = x(uvt);
     bool invalidU = any2(uvt < VEC2(0.f));
     bool invalidV = any2((VEC2(uvt) + (eql(0) ? 0.f : vd)) > VEC2(1.f));

    if (invalidU || invalidV) valid = false;
    if (ibs(valid)) return INFINITY;

    UV.xy = compvec2(uvt);
     float dst = z(uvt);
    return (lessF(dst, 0.0f) || !valid) ? INFINITY : dst;
}

TResult choiceFirstBaked(inout TResult res) {
     int tri = exchange(bakedRange[0], LONGEST);

     bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;
        
    if (ibs(validTriangle)) return res;

     vec2 uv = bakedRangeIntersection[0].yz;
     float _d = bakedRangeIntersection[0].x;
     bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

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

     int tri = tpi < exchange(bakedStackCount, 0) ? bakedStack[tpi] : LONGEST;

     bool validTriangle = 
        tri >= 0 && 
        tri != LONGEST;

    vec3 triverts = gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri)), lane4).wzx;

    vec2 uv = vec2(0.0f);
     float _d = intersectTriangle(orig, dir, triverts, uv);
     bool near = validTriangle && lessF(_d, INFINITY) && lessEqualF(_d, res.dist);

    if (near) {
        res.dist = _d;
        res.triangle = tri;
        res.uv.xy = uv;
    }
    
    return res;
}

TResult testIntersection(inout TResult res, in VEC3 orig, in VEC3 dir, in int tri, in int step, in bool isValid) {
     bool validTriangle = 
        isValid && 
        tri >= 0 && 
        tri != LONGEST;

    vec3 triverts = gatherMosaicCompDyn(vertex_texture, gatherMosaic(getUniformCoord(tri)), lane4).wzx;

    vec2 uv = vec2(0.0f);
     float _d = intersectTriangle(orig, dir, triverts, uv);
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
     VEC4 nps = mult4w(eql(3) ? 1.0f : orig, GEOMETRY_BLOCK octreeUniform.project);
    return nps / w(nps);
}

VEC3 unprojectVoxels(in VEC3 orig) {
     VEC4 nps = mult4w(eql(3) ? 1.0f : orig, GEOMETRY_BLOCK octreeUniform.unproject);
    return nps / w(nps);
}

float intersectCubeSingle(in VEC3 origin, in VEC3 ray, in VEC3 cubeMin, in VEC3 cubeMax, inout float near, inout float far) {
     VEC3 dr = 1.0f / ray;
     VEC3 tMin = (cubeMin - origin) * dr;
     VEC3 tMax = (cubeMax - origin) * dr;
     VEC3 t1 = min(tMin, tMax);
     VEC3 t2 = max(tMin, tMax);
     float tNear = compmax3(t1);
     float tFar  = compmin3(t2);
     bool isCube = tFar >= tNear && greaterEqualF(tFar, 0.0f);
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}




// optimized by WARP and SIMD intersection boxes

vec2 intersectCubeSingleApart(in VEC3 origin, in VEC3 ray, in VEC3 cubeMin, in VEC3 cubeMax) {
     VEC3 dr = 1.0f / ray;
     VEC3 tMin = (cubeMin - origin) * dr;
     VEC3 tMax = (cubeMax - origin) * dr;
     VEC3 t1 = min(tMin, tMax);
     VEC3 t2 = max(tMin, tMax);
    return vec2(compmax3(t1), compmin3(t2));
}

float calculateRealDistance(in float tNear, in float tFar, inout float near, inout float far) {
     bool isCube = tFar >= tNear && greaterEqualF(tFar, 0.0f);
    near = isCube ? tNear : INFINITY;
    far  = isCube ? tFar  : INFINITY;
    return isCube ? (lessF(tNear, 0.0f) ? tFar : tNear) : INFINITY;
}



const VEC3 padding = VEC3(0.00001f);
const int STACK_SIZE = 16;
shared int deferredStack[WORK_SIZE][STACK_SIZE];
shared int deferredPtr[WORK_SIZE];
shared int idx[WORK_SIZE];
shared bool skip[WORK_SIZE];
shared bool validBox[WORK_SIZE];

//TResult traverse(in float distn, in VEC3 origin, in VEC3 direct, in Hit hit) {
TResult traverse(in float distn, in VEC3 origin, in VEC3 direct, in Hit hit) {
    const uint L = invoc(gl_LocalInvocationID.x);
     //VEC4 origin = eql(3) ? 1.0f : swiz(_origin); 
     //VEC4 direct = eql(3) ? 0.0f : swiz(_direct);

    TResult lastRes;
    lastRes.dist = INFINITY;
    lastRes.predist = INFINITY;
    lastRes.triangle = LONGEST;
    lastRes.materialID = LONGEST;
    bakedRange[0] = LONGEST;

    // test constants
     int bakedStep = int(floor(1.f + hit.vmods.w));
     VEC4 torig = projectVoxels(origin);
     VEC4 tdirproj = mult4w(direct, GEOMETRY_BLOCK octreeUniform.project);
     float dirlen = 1.0f / length3(tdirproj);
     VEC4 dirproj = normalize3(tdirproj);
    
     vec3 torigUnif = compvec3(torig);
     vec3 dirprojUnif = compvec3(dirproj);

    // test with root node
    //HlbvhNode node = Nodes[idx];
    // bbox lbox = node.box;
    float near = INFINITY, far = INFINITY;
    // float d = intersectCubeSingle(torig, dirproj, lbox.mn.xyz, lbox.mx.xyz, near, far);
     float d = intersectCubeSingle(torig, dirproj, VEC3(0.0f), VEC3(1.0f), near, far);
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

         bool isLeaf = node.pdata.x == node.pdata.y && validBox[L];
        if (bs(isLeaf)) {
            testIntersection(lastRes, origin, direct, node.pdata.w, bakedStep, isLeaf);
        }

         bool notLeaf = node.pdata.x != node.pdata.y && validBox[L];
        if (bs(notLeaf)) {

            // search intersection worklets (3 lanes occupy)
            vec2 hfn_leftright[2] = {
                 intersectCubeSingleApart(torig, dirproj, Nodes[node.pdata.x].box.mn[lane4], Nodes[node.pdata.x].box.mx[lane4]),
                 intersectCubeSingleApart(torig, dirproj, Nodes[node.pdata.y].box.mn[lane4], Nodes[node.pdata.y].box.mx[lane4])
            };

            // determine parallel (2 lanes occupy)
            float nearLR = INFINITY, farLR = INFINITY;
             int distrb = eql(0) ? 0 : 1;
             float leftrighthit = calculateRealDistance(hfn_leftright[distrb].x, hfn_leftright[distrb].y, nearLR, farLR);
             bool overlapsVc = notLeaf && lessF(leftrighthit, INFINITY) && greaterEqualF(leftrighthit, 0.0f) && greaterEqualF(lastRes.predist, near * dirlen);
            
            // compose results
             bvec2 overlaps = compbvec2(overlapsVc);
             bool overlapAny = any(overlaps);
            if (bs(overlapAny)) {
                 bool leftNearb = lessEqualF(x(leftrighthit), y(leftrighthit));
                if (overlapAny && mt()) {
                     bool leftOrder = all(overlaps) ? leftNearb : overlaps.x;
                    ivec2 leftright = mix(ivec2(-1), node.pdata.xy, overlaps);
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
                 int ptr = --deferredPtr[L];
                 bool valid = ptr >= 0;
                idx[L] = valid ? exchange(deferredStack[L][ptr], -1) : -1;
                validBox[L] = validBox[L] && valid && idx[L] >= 0;
            } skip[L] = false;
        }
    }

    choiceBaked(lastRes, origin, direct, bakedStep);
    return loadInfo(lastRes);
}
