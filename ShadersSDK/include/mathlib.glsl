
// Readme license https://github.com/AwokenGraphics/prismarine-core/blob/master/LICENSE.md


#ifndef _STOMATH_H
#define _STOMATH_H


// roundly comparsion functions
bool lessEqualF(in float a, in float b) { return (b-a) > -PZERO; }
bool lessF(in float a, in float b) { return (b-a) >= PZERO; }
bool greaterEqualF(in float a, in float b) { return (a-b) > -PZERO; }
bool greaterF(in float a, in float b) { return (a-b) >= PZERO; }
bool equalF(in float a, in float b) { return abs(a-b) < PZERO; }


// vector math utils
float sqlen(in vec3 a) { return dot(a, a); }
float sqlen(in vec2 a) { return dot(a, a); }
float sqlen(in float v) { return v * v; }
float mlength(in vec3 mcolor){ return max(mcolor.x, max(mcolor.y, mcolor.z)); }
vec4 divW(in vec4 aw){ return aw / aw.w; }


// memory managment
void swap(inout int a, inout int b){ int t = a; a = b; b = t; }
uint exchange(inout uint mem, in uint v){ uint tmp = mem; mem = v; return tmp; }
int exchange(inout int mem, in int v){ int tmp = mem; mem = v; return tmp; }
int add(inout int mem, in int v){ int tmp = mem; mem += v; return tmp; }

// logical functions
bvec2 not2(in bvec2 a) { return bvec2(!a.x, !a.y); }
bvec2 and2(in bvec2 a, in bvec2 b) { return bvec2(a.x && b.x, a.y && b.y); }
bvec2 or2(in bvec2 a, in bvec2 b) { return bvec2(a.x || b.x, a.y || b.y); }

// logical functions (bvec4)
bvec4 or(in bvec4 a, in bvec4 b){
    return bvec4(
        a.x || b.x,
        a.y || b.y,
        a.z || b.z,
        a.w || b.w
    );
}

bvec4 and(in bvec4 a, in bvec4 b){
    return bvec4(
        a.x && b.x,
        a.y && b.y,
        a.z && b.z,
        a.w && b.w
    );
}

bvec4 not(in bvec4 a){
    return bvec4(!a.x, !a.y, !a.z, !a.w);
}




// mixing functions
void mixed(inout vec3 src, inout vec3 dst, in float coef){ dst *= coef; src *= 1.0f - coef; }
void mixed(inout vec3 src, inout vec3 dst, in vec3 coef){ dst *= coef; src *= 1.0f - coef; }


// matrix math
vec4 mult4(in vec4 vec, in mat4 tmat){
    //return vec4(dot(tmat[0], vec), dot(tmat[1], vec), dot(tmat[2], vec), dot(tmat[3], vec));
    return vec * tmat;
}

vec4 mult4(in mat4 tmat, in vec4 vec){
    //return fma(tmat[0], vec.xxxx, fma(tmat[1], vec.yyyy, fma(tmat[2], vec.zzzz, tmat[3] * vec.wwww)));
    return tmat * vec;
}


int modi(in int a, in int b){
    return (a % b + b) % b;
}

uint btc(in uint vlc){
    return vlc == 0 ? 0 : bitCount(vlc);
}





float intersectCubeSingle(in vec3 origin, in vec3 ray, in vec4 cubeMin, in vec4 cubeMax, inout float near, inout float far) {
    vec3 dr = 1.0f / ray;
    vec3 norig = -origin*dr;
    vec3 tMin = fma(cubeMin.xyz, dr, norig);
    vec3 tMax = fma(cubeMax.xyz, dr, norig);
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
#ifdef ENABLE_AMD_INSTRUCTION_SET
    float tNear = max3(t1.x, t1.y, t1.z);
    float tFar  = min3(t2.x, t2.y, t2.z);
#else
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);
#endif
    bool isCube = greaterEqualF(tFar, tNear) && greaterEqualF(tFar, 0.0f);
    float inf = INFINITY;
    near = isCube ? min(tNear, tFar) : inf;
    far  = isCube ? max(tNear, tFar) : inf;
    return (isCube ? (lessF(near, 0.0f) ? far : near) : inf);
}


vec2 intersectCubeDual(
    in vec3 origin, in vec3 ray, 
    in mat2x4 cubeMin, in mat2x4 cubeMax,
    inout vec2 near, inout vec2 far
) {
    vec3 dr = 1.0f / ray;

    mat3x2 dr2 = mat3x2(dr.xx, dr.yy, dr.zz);
    mat3x2 origin2 = mat3x2(origin.xx, origin.yy, origin.zz);
    mat4x2 cubeMin2 = transpose(cubeMin);
    mat4x2 cubeMax2 = transpose(cubeMax);

    mat3x2 norig = mat3x2(-origin2[0]*dr2[0], -origin2[1]*dr2[1], -origin2[2]*dr2[2]);
    mat3x2 tMin = mat3x2(
        fma(cubeMin2[0], dr2[0], norig[0]), 
        fma(cubeMin2[1], dr2[1], norig[1]), 
        fma(cubeMin2[2], dr2[2], norig[2])
    );
    mat3x2 tMax = mat3x2(
        fma(cubeMax2[0], dr2[0], norig[0]), 
        fma(cubeMax2[1], dr2[1], norig[1]), 
        fma(cubeMax2[2], dr2[2], norig[2])
    );

    mat3x2 t1 = mat3x2(min(tMin[0], tMax[0]), min(tMin[1], tMax[1]), min(tMin[2], tMax[2]));
    mat3x2 t2 = mat3x2(max(tMin[0], tMax[0]), max(tMin[1], tMax[1]), max(tMin[2], tMax[2]));
#ifdef ENABLE_AMD_INSTRUCTION_SET
    vec2 tNear = max3(t1[0], t1[1], t1[2]);
    vec2 tFar  = min3(t2[0], t2[1], t2[2]);
#else
    vec2 tNear = max(max(t1[0], t1[1]), t1[2]);
    vec2 tFar  = min(min(t2[0], t2[1]), t2[2]);
#endif
    vec2 inf = vec2(INFINITY);
    bvec2 isCube = and2(greaterThanEqual(tFar+PZERO, tNear), greaterThanEqual(tFar+PZERO, vec2(0.0f)));
    near = mix(inf, min(tNear, tFar), isCube);
    far  = mix(inf, max(tNear, tFar), isCube);
    return mix(near, far, lessThanEqual(near + PZERO, vec2(0.0f)));
}






#define BFE(a,o,n) ((a >> o) & ((1 << n)-1))

uvec2 U2P(in uint64_t pckg) {
    return uvec2((pckg >> 0) & 0xFFFFFFFF, (pckg >> 32) & 0xFFFFFFFF);
}

int BFI(in int base, in int inserts, in int offset, in int bits){
    int mask = bits >= 32 ? 0xFFFFFFFF : (1<<bits)-1;
    int offsetMask = mask << offset;
    return ((base & (~offsetMask)) | ((inserts & mask) << offset));
}




// bit logic
const int bx = 1, by = 2, bz = 4, bw = 8;
const int bswiz[8] = {1, 2, 4, 8, 16, 32, 64, 128};

int cB(in bool a, in int swizzle){
    return a ? swizzle : 0;
}


int cB4(in bvec4 a){
    ivec4 mx4 = mix(ivec4(0), ivec4(bx, by, bz, bw), a);
    ivec2 mx2 = mx4.xy | mx4.wz; // swizzle or
    return (mx2.x | mx2.y); // rest of
}

int cB2(in bvec2 a){
    ivec2 mx = mix(ivec2(0), ivec2(bx, by), a);
    return (mx.x | mx.y);
}

bvec2 cI2(in int a){
    return bvec2(
        (a & bx) > 0, 
        (a & by) > 0
    );
}

bool anyB(in int a){
    return a > 0;
}

bool allB2(in int a){
    return a == 3;
}




vec4 cubic(in float v){
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

vec4 textureBicubic(in sampler2D tx, in vec2 texCoords) {
    vec2 texSize = textureSize(tx, 0);
    vec2 invTexSize = 1.0f / texSize;

    texCoords *= texSize;
    vec2 fxy = fract(texCoords);
    texCoords = floor(texCoords);

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2(0.0f, 1.0f).xyxy;
    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = vec4(c + vec4(xcubic.yw, ycubic.yw) / s) * invTexSize.xxyy;

    vec4 sample0 = texture(tx, offset.xz);
    vec4 sample1 = texture(tx, offset.yz);
    vec4 sample2 = texture(tx, offset.xw);
    vec4 sample3 = texture(tx, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}



vec4 unpackHalf(in uvec2 halfs){
    return vec4(unpackHalf2x16(halfs.x), unpackHalf2x16(halfs.y));
}

uvec2 packHalf(in vec4 floats){
    return uvec2(packHalf2x16(floats.xy), packHalf2x16(floats.zw));
}


// reserved for future rasterizers
// just for save
vec3 barycentric2D(in vec3 p, in mat3x3 triangle){
    mat3x3 plc = transpose(mat3x3(triangle[2] - triangle[0], triangle[1] - triangle[0], triangle[0] - p));
    vec3 u = cross(plc[0], plc[1]); // xy (2d) cross
    if (abs(u.z) < 1.f) return vec3(-1.f,1.f,1.f); 
    return vec3(u.z-(u.x+u.y), u.y, u.x)/u.z;
}



#endif