// get relative invocation ID (active)
uint activeInvocation() {
    const uint64_t activeBits = ballotARB(true) & gl_SubGroupLtMaskARB;
    const uvec2 unpacked = unpackUint2x32(activeBits);
    return (bitCount(unpacked.x) + bitCount(unpacked.y));
    //return gl_SubGroupInvocationARB;
}



// get grouped swizzle
vec4 swiz4(in vec4 vc) {
    const uint sz = activeInvocation() % 4;
    if (sz == 1) return vc.yyyy; else 
    if (sz == 2) return vc.zzzz; else 
    if (sz == 3) return vc.wwww; 
    return vc.xxxx;
}


// get swizzle component from packed vectors
float swiz(in vec4 vc) {
    const uint sz = activeInvocation() % 4;
    if (sz == 1) return float(vc.y); else 
    if (sz == 2) return float(vc.z); else 
    if (sz == 3) return float(vc.w);
    return float(vc.x);
}

float swiz(in vec3 vc) {
    const uint sz = activeInvocation() % 4;
    if (sz == 1) return float(vc.y); else 
    if (sz == 2) return float(vc.z);
    return float(vc.x);
}

float swiz(in vec2 vc) {
    const uint sz = activeInvocation() % 4;
    if (sz == 1) return float(vc.y);
    return float(vc.x);
}

int swiz(in ivec2 vc) {
    const uint sz = activeInvocation() % 4;
    if (sz == 1) return int(vc.y);
    return int(vc.x);
}

bool swiz(in bvec2 vc) {
    const uint sz = activeInvocation() % 4;
    if (sz == 1) return bool(vc.y);
    return bool(vc.x);
}


// read lane from vector4
float lane(in float mem, in uint l) {
    return readInvocationARB(mem, (activeInvocation() / 4) * 4 + (l % 4));
}

int lane(in int mem, in uint l) {
    return readInvocationARB(mem, (activeInvocation() / 4) * 4 + (l % 4));
}

uint lane(in uint mem, in uint l) {
    return readInvocationARB(mem, (activeInvocation() / 4) * 4 + (l % 4));
}

bool lane(in bool mem, in uint l) {
    return bool(readInvocationARB(uint(mem), (activeInvocation() / 4) * 4 + (l % 4)));
}


// defined lanes
float x(in float mem) { return lane(mem, 0); }
float y(in float mem) { return lane(mem, 1); }
float z(in float mem) { return lane(mem, 2); }
float w(in float mem) { return lane(mem, 3); }

// int lanes
int x(in int mem) { return lane(mem, 0); }
int y(in int mem) { return lane(mem, 1); }
int z(in int mem) { return lane(mem, 2); }
int w(in int mem) { return lane(mem, 3); }

// uint lanes
uint x(in uint mem) { return lane(mem, 0); }
uint y(in uint mem) { return lane(mem, 1); }
uint z(in uint mem) { return lane(mem, 2); }
uint w(in uint mem) { return lane(mem, 3); }

// bool lanes
bool x(in bool mem) { return lane(mem, 0); }
bool y(in bool mem) { return lane(mem, 1); }
bool z(in bool mem) { return lane(mem, 2); }
bool w(in bool mem) { return lane(mem, 3); }


// shifted getter
float sw1(in float mem){
    return lane(mem, (activeInvocation() % 4) + 1);
}

float sw2(in float mem){
    return lane(mem, (activeInvocation() % 4) + 2);
}

float sw3(in float mem){
    return lane(mem, (activeInvocation() % 4) + 3);
}


// swap lanes
float swapXY(in float mem){
    return (activeInvocation() % 2) == 1 ? x(mem) : y(mem);
}

int swapXY(in int mem){
    return (activeInvocation() % 2) == 1 ? x(mem) : y(mem);
}

uint swapXY(in uint mem){
    return (activeInvocation() % 2) == 1 ? x(mem) : y(mem);
}


// compact vector
vec4 compvec4(in float mem){
    return vec4(x(mem), y(mem), z(mem), w(mem));
}

vec3 compvec3(in float mem){
    return vec3(x(mem), y(mem), z(mem));
}

vec2 compvec2(in float mem){
    return vec2(x(mem), y(mem));
}

ivec2 compivec(in int mem){
    return ivec2(x(mem), y(mem));
}

uvec2 compuvec(in int mem){
    return uvec2(x(mem), y(mem));
}

bvec2 compbvec(in bool mem){
    return bvec2(x(mem), y(mem));
}


// get length of vector 3 (optimized)
float length3(in float a){
    const vec3 vc = compvec3(a);
    return sqrt(dot(vc,vc));
}


// dot product between lanes
float dot2(in float a, in float b){
    return dot(compvec2(a), compvec2(b));
}

float dot3(in float a, in float b){
    return dot(compvec3(a), compvec3(b));
}

float dot4(in float a, in float b){
    return dot(compvec4(a), compvec4(b));
}


// matrix math on WARP%4 lanes (require compacted vector)
float mult4w(in vec4 vec, in mat4 mat){
    return dot(mat[gl_SubGroupInvocationARB % 4], vec);
}

float mult4w(in mat4 mat, in vec4 vec){
    return dot( vec4(swiz(mat[0]), swiz(mat[1]), swiz(mat[2]), swiz(mat[3])), vec );
}


// matrix math on WARP%4 lanes (laned version)
float mult4w(in float vec, in mat4 mat){
    return mult4w(compvec4(vec), mat);
}

float mult4w(in mat4 mat, in float vec){
    return mult4w(mat, compvec4(vec));
}


// is work lane (for most operations)
bool mt(){
    return (activeInvocation() % 4) == 0;
}


// odd even lanes
bool oddl(){
    return (activeInvocation() % 2) == 1;
}

bool evenl(){
    return (activeInvocation() % 2) == 0;
}


// compare lanes (for vector operations)
bool lessl(in int lane){
    return (activeInvocation() % 4) < lane;
}

bool eql(in int lane){
    return (activeInvocation() % 4) == lane;
}

bool lessql(in int lane){
    return (activeInvocation() % 4) <= lane;
}


// boolean voting
bool any2(in bool bl){
    return lane(bl, 0) || lane(bl, 1);
}

bool all2(in bool bl){
    return lane(bl, 0) && lane(bl, 1);
}

bool any3(in bool bl){
    return lane(bl, 0) || lane(bl, 1) || lane(bl, 2);
}

bool all3(in bool bl){
    return lane(bl, 0) && lane(bl, 1) && lane(bl, 2);
}


// divide invocations
uint invoc(in uint inv){
    return inv / 4;
}

int invoc(in int inv){
    return inv / 4;
}


// cross lane "cross product"
float cross3(in float a, in float b){
    const uint ln = activeInvocation() % 3;
    return dot(vec2(
        lane(a, (ln + 1) % 3),
        lane(b, (ln + 1) % 3)
    ), -vec2(
        lane(b, (ln + 2) % 3),
        lane(a, (ln + 2) % 3)
    ));
}


// warp based types
#define VEC4 float
#define VEC3 float
#define VEC2 float
#define IVEC4 int
#define IVEC3 int
#define IVEC2 int
#define UVEC4 uint
#define UVEC3 uint
#define UVEC2 uint
#define BVEC4 bool
#define BVEC3 bool
#define BVEC2 bool


