
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

#ifdef ENABLE_AMD_INSTRUCTION_SET
float mlength(in vec3 mcolor){ return max3(mcolor.x, mcolor.y, mcolor.z); }
#else
float mlength(in vec3 mcolor){ return max(mcolor.x, max(mcolor.y, mcolor.z)); }
#endif
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
    //return fma(tmat[0], vec.xxxx, fma(tmat[1], vec.yyyy, fma(tmat[2], vec.zzzz, tmat[3] * vec.wwww)));
    return tmat * vec;
}

vec4 mult4(in mat4 tmat, in vec4 vec){
    //return vec4(dot(tmat[0], vec), dot(tmat[1], vec), dot(tmat[2], vec), dot(tmat[3], vec));
    return vec * tmat;
}


int modi(in int a, in int b){
    return (a % b + b) % b;
}

uint btc(in uint vlc){
    return vlc == 0 ? 0 : bitCount(vlc);
    //return bitCount(vlc);
}

int lsb(in uint vlc){
    return vlc == 0 ? -1 : findLSB(vlc);
    //return findLSB(vlc);
}

int msb(in uint vlc){
    return vlc == 0 ? -1 : findMSB(vlc);
    //return findMSB(vlc);
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


#ifdef AMD_F16_BVH
vec2 intersectCubeDual(
    in f16vec3 origin, in f16vec3 dr, 
    in f16mat2x4 cubeMin, in f16mat2x4 cubeMax,
    //in mat2x4 cubeMin, in mat2x4 cubeMax,
    inout vec2 near, inout vec2 far
) 
#else
vec2 intersectCubeDual(
    in vec3 origin, in vec3 dr, 
    in mat2x4 cubeMin, in mat2x4 cubeMax,
    inout vec2 near, inout vec2 far
) 
#endif
{
#ifdef AMD_F16_BVH
    f16mat3x4 dr2 = f16mat3x4(dr.xxxx, dr.yyyy, dr.zzzz);
    f16mat3x4 origin2 = f16mat3x4(origin.xxxx, origin.yyyy, origin.zzzz);
    f16mat4x4 cubeMinMax2 = transpose(f16mat4x4(cubeMin[0], cubeMin[1], cubeMax[0], cubeMax[1]));

    f16mat3x4 norig = f16mat3x4(-origin2[0]*dr2[0], -origin2[1]*dr2[1], -origin2[2]*dr2[2]);
    f16mat3x4 tMinMax = f16mat3x4(
        fma(cubeMinMax2[0], dr2[0], norig[0]), 
        fma(cubeMinMax2[1], dr2[1], norig[1]), 
        fma(cubeMinMax2[2], dr2[2], norig[2])
    );

    f16mat3x2 t1 = f16mat3x2(min(tMinMax[0].xy, tMinMax[0].zw), min(tMinMax[1].xy, tMinMax[1].zw), min(tMinMax[2].xy, tMinMax[2].zw));
    f16mat3x2 t2 = f16mat3x2(max(tMinMax[0].xy, tMinMax[0].zw), max(tMinMax[1].xy, tMinMax[1].zw), max(tMinMax[2].xy, tMinMax[2].zw));
#ifdef ENABLE_AMD_INSTRUCTION_SET
    f16vec2 tNear = max3(t1[0], t1[1], t1[2]);
    f16vec2 tFar  = min3(t2[0], t2[1], t2[2]);
#else
    f16vec2 tNear = max(max(t1[0], t1[1]), t1[2]);
    f16vec2 tFar  = min(min(t2[0], t2[1]), t2[2]);
#endif
#else 
    mat3x4 dr2 = mat3x4(dr.xxxx, dr.yyyy, dr.zzzz);
    mat3x4 origin2 = mat3x4(origin.xxxx, origin.yyyy, origin.zzzz);
    mat4x4 cubeMinMax2 = transpose(mat4x4(cubeMin[0], cubeMin[1], cubeMax[0], cubeMax[1]));

    mat3x4 norig = mat3x4(-origin2[0]*dr2[0], -origin2[1]*dr2[1], -origin2[2]*dr2[2]);
    mat3x4 tMinMax = mat3x4(
        fma(cubeMinMax2[0], dr2[0], norig[0]), 
        fma(cubeMinMax2[1], dr2[1], norig[1]), 
        fma(cubeMinMax2[2], dr2[2], norig[2])
    );

    mat3x2 t1 = mat3x2(min(tMinMax[0].xy, tMinMax[0].zw), min(tMinMax[1].xy, tMinMax[1].zw), min(tMinMax[2].xy, tMinMax[2].zw));
    mat3x2 t2 = mat3x2(max(tMinMax[0].xy, tMinMax[0].zw), max(tMinMax[1].xy, tMinMax[1].zw), max(tMinMax[2].xy, tMinMax[2].zw));
#ifdef ENABLE_AMD_INSTRUCTION_SET
    vec2 tNear = max3(t1[0], t1[1], t1[2]);
    vec2 tFar  = min3(t2[0], t2[1], t2[2]);
#else
    vec2 tNear = max(max(t1[0], t1[1]), t1[2]);
    vec2 tFar  = min(min(t2[0], t2[1]), t2[2]);
#endif
#endif

    vec2 inf = vec2(INFINITY);
    bvec2 isCube = and2(greaterThanEqual(tFar+PZERO, tNear), greaterThanEqual(tFar+PZERO, vec2(0.0f)));
    near = mix(inf, vec2(min(tNear, tFar)), isCube);
    far  = mix(inf, vec2(max(tNear, tFar)), isCube);
    return mix(near, far, lessThanEqual(near + PZERO, vec2(0.0f)));
}






//#define BFE(a,o,n) ((a >> o) & ((1 << n)-1))
//#define BFE_HW(a,o,n) (bitfieldExtract(a,o,n))

uvec2 U2P(in uint64_t pckg) {
    //return uvec2((pckg >> 0) & 0xFFFFFFFF, (pckg >> 32) & 0xFFFFFFFF);
    return unpackUint2x32(pckg);
}

int BFE(in int base, in int offset, in int bits){
    return (base >> offset) & ((1 << bits)-1);
    //int result = bitfieldExtract(base, offset, bits);
    //return result;
    //return bitfieldExtract(base, offset, bits);
}

int BFE_HW(in int base, in int offset, in int bits){
    return bitfieldExtract(base, offset, bits);
}


int BFE(in uint base, in int offset, in int bits){
    //return int((base >> offset) & ((1 << bits)-1));
    //int result = int(bitfieldExtract(base, offset, bits));
    //return result;
    return int(bitfieldExtract(base, offset, bits));
}

int BFE_HW(in uint base, in int offset, in int bits){
    return int(bitfieldExtract(base, offset, bits));
}



int BFI(in int base, in int inserts, in int offset, in int bits){
    //int mask = bits >= 32 ? 0xFFFFFFFF : (1<<bits)-1;
    //int offsetMask = mask << offset;
    //return ((base & (~offsetMask)) | ((inserts & mask) << offset));
    return bitfieldInsert(base, inserts, offset, bits);
}

int BFI_HW(in int base, in int inserts, in int offset, in int bits){
    return bitfieldInsert(base, inserts, offset, bits);
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



mediump vec4 unpackHalf(in uint64_t halfs){
    uvec2 hilo = unpackUint2x32(halfs);
    return vec4(unpackHalf2x16(hilo.x), unpackHalf2x16(hilo.y));
}

mediump vec4 unpackHalf(in uvec2 hilo){
    return vec4(unpackHalf2x16(hilo.x), unpackHalf2x16(hilo.y));
}

uvec2 packHalf2(in vec4 floats){
    return (uvec2(packHalf2x16(floats.xy), packHalf2x16(floats.zw)));
}

uint64_t packHalf(in vec4 floats){
    return packUint2x32(uvec2(packHalf2x16(floats.xy), packHalf2x16(floats.zw)));
}


#ifdef ENABLE_AMD_INSTRUCTION_SET
f16vec4 unpackHalf2(in uint64_t halfs){
    uvec2 hilo = unpackUint2x32(halfs);
    //return f16vec4(unpackFloat2x16(hilo.x), unpackFloat2x16(hilo.y));
    return f16vec4(unpackHalf(halfs));
}

f16vec4 unpackHalf2(in uvec2 hilo){
    //return f16vec4(unpackFloat2x16(hilo.x), unpackFloat2x16(hilo.y));
    return f16vec4(unpackHalf(hilo));
}

uvec2 packHalf2(in f16vec4 floats){
    //return (uvec2(packFloat2x16(floats.xy), packFloat2x16(floats.zw)));
    return (uvec2(packHalf2x16(floats.xy), packHalf2x16(floats.zw)));
}

uint64_t packHalf(in f16vec4 floats){
    //return packUint2x32(uvec2(packFloat2x16(floats.xy), packFloat2x16(floats.zw)));
    return packUint2x32(uvec2(packHalf2x16(floats.xy), packHalf2x16(floats.zw)));
}
#endif


// reserved for future rasterizers
// just for save
vec3 barycentric2D(in vec3 p, in mat3x3 triangle){
    mat3x3 plc = transpose(mat3x3(triangle[2] - triangle[0], triangle[1] - triangle[0], triangle[0] - p));
    vec3 u = cross(plc[0], plc[1]); // xy (2d) cross
    if (abs(u.z) < 1.f) return vec3(-1.f,1.f,1.f); 
    return vec3(u.z-(u.x+u.y), u.y, u.x)/u.z;
}



#endif