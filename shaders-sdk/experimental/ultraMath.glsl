// get relative invocation ID (active)
uint activeInvocation() {
    // uint64_t activeBits = ballotARB(true) & gl_SubGroupLtMaskARB;
    // uvec2 unpacked = unpackUint2x32(activeBits);
    //return (bitCount(unpacked.x) + bitCount(unpacked.y));
    return gl_SubGroupInvocationARB;
}

const uint SCALARS = 4; // use between 4 or 8 in NVidia GPU
uint laneP = (gl_SubGroupInvocationARB%SCALARS);
uint offtP = (gl_SubGroupInvocationARB/SCALARS)*SCALARS;

uint lane4 = (gl_SubGroupInvocationARB&3);
uint offt4 = (gl_SubGroupInvocationARB>>2)<<2;

#define sz lane4 // redirect

// get grouped swizzle
vec4 swiz4(in vec4 _vc) {
     vec4 vc = readInvocationARB(_vc, offt4);
    // uint sz = lane4;
    return ((sz == 1) ? vc.yyyy : ((sz == 2) ? vc.zzzz : ((sz == 3) ? vc.wwww : vc.xxxx)));
}


// get swizzle component from packed vectors
float swiz(in vec4 _vc) {
     vec4 vc = readInvocationARB(_vc, offt4);
    // uint sz = lane4;
    return ((sz == 1) ? vc.y : ((sz == 2) ? vc.z : ((sz == 3) ? vc.w : vc.x)));
}

float swiz(in vec3 _vc) {
     vec3 vc = readInvocationARB(_vc, offt4);
    // uint sz = lane4;
    return ((sz == 1) ? vc.y : ((sz == 2) ? vc.z : vc.x));
}

float swiz(in vec2 _vc) {
     vec2 vc = readInvocationARB(_vc, offt4);
    // uint sz = lane4;
    return ((sz == 1) ? vc.y : vc.x);
}

int swiz(in ivec2 _vc) {
     ivec2 vc = readInvocationARB(_vc, offt4);
    // uint sz = lane4;
    return ((sz == 1) ? vc.y : vc.x);
}

bool swiz(in bvec2 _vc) {
     bvec2 vc = bvec2(readInvocationARB(uvec2(_vc), offt4));
    // uint sz = lane4;
    return ((sz == 1) ? vc.y : vc.x);
}


// read lane from vector4
float lane(in float mem, in uint l) {
    return readInvocationARB(mem, offtP + l);
}

vec2 lane(in vec2 mem, in uint l) {
    return readInvocationARB(mem, offtP + l);
}

int lane(in int mem, in uint l) {
    return readInvocationARB(mem, offtP + l);
}

uint lane(in uint mem, in uint l) {
    return readInvocationARB(mem, offtP + l);
}

bool lane(in bool mem, in uint l) {
    return bool(readInvocationARB(uint(mem), offtP + l));
}


// defined lanes
float x(in float mem) { return lane(mem, 0); }
float y(in float mem) { return lane(mem, 1); }
float z(in float mem) { return lane(mem, 2); }
float w(in float mem) { return lane(mem, 3); }

// vec2 lanes
vec2 x(in vec2 mem) { return lane(mem, 0); }
vec2 y(in vec2 mem) { return lane(mem, 1); }
vec2 z(in vec2 mem) { return lane(mem, 2); }
vec2 w(in vec2 mem) { return lane(mem, 3); }

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


// swap lanes
float swapXY(in float mem){
     float _x = x(mem);
     float _y = y(mem);
    return laneP == 1 ? _x : _y;
}

int swapXY(in int mem){
     int _x = x(mem);
     int _y = y(mem);
    return laneP == 1 ? _x : _y;
}

uint swapXY(in uint mem){
     uint _x = x(mem);
     uint _y = y(mem);
    return laneP == 1 ? _x : _y;
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

ivec2 compivec2(in int mem){
    return ivec2(x(mem), y(mem));
}

uvec2 compuvec2(in uint mem){
    return uvec2(x(mem), y(mem));
}

bvec2 compbvec2(in bool mem){
    return bvec2(x(mem), y(mem));
}





// is work lane (for most operations)
bool mt(){
    return laneP == 0;
}


// odd even lanes
bool oddl(){
    return (laneP&1) == 1;
}

bool evenl(){
    return (laneP&1) == 0;
}


// compare lanes (for vector operations)
bool lessl(in int lane){
    return laneP < lane;
}

bool eql(in int lane){
    return laneP == lane;
}

bool lessql(in int lane){
    return laneP <= lane;
}



// dot product between lanes, uses basic reduction
float dot2(in float a, in float b) { // generally, only 2 ops only
    //return dot(compvec2(a), compvec2(b));
     float c = a * b;
    return x(c) + y(c);
}

float dot3(in float a, in float b) { // generally, only 3 ops only
    //return dot(compvec3(a), compvec3(b));
     float c = a * b;
     float pl = lessl(2) ? (x(c) + y(c)) : (z(c) + 0.0f);
    return x(pl) + z(pl);
}

float dot4(in float a, in float b) { // generally, only 3 ops only
    //return dot(compvec4(a), compvec4(b));
     float c = a * b;
     float pl = lessl(2) ? (x(c) + y(c)) : (z(c) + w(c));
    return x(pl) + z(pl);
}


// matrix math on WARP%4 lanes (laned version)
float mult4w(in float vec, in mat4 mat){
    return dot(mat[lane4], compvec4(vec));
}

float mult4w(in mat4 mat, in float vec){
    return dot(vec4(swiz(mat[0]), swiz(mat[1]), swiz(mat[2]), swiz(mat[3])), compvec4(vec));
}


// get length of vector 3 (optimized)
float length3(in float a){
    return sqrt(dot3(a, a));//length(compvec3(a));
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
    return inv / SCALARS;
}

int invoc(in int inv){
    return inv / int(SCALARS);
}

// constants bank of cross lanes
 uint cslnsX[4] = {1, 2, 0, 3};
 uint cslnsY[4] = {2, 0, 1, 3};

// cross lane "cross product"
float cross3(in float a, in float b){
     uint ln = laneP&3;
    return dot(vec2(
         lane(a, cslnsX[ln]), 
         lane(b, cslnsX[ln])
    ), vec2(
         lane(b, cslnsY[ln]), 
        -lane(a, cslnsY[ln])
    ));
}


// normalize function for warp
float normalize3(in float a){
    return a / length3(a);
}


void putv3(in float inp, inout vec3 mem, inout int loc){
    if (loc == 0) mem.x = inp; else
    if (loc == 1) mem.y = inp; else
    if (loc == 2) mem.z = inp;
}

// if any valid
bool bs(in bool valid) {
    return anyInvocationARB(valid && mt());
}

// if all invalid
bool ibs(in bool valid) {
    return !anyInvocationARB(valid && mt());
}


float compmax3(in float t1){
#ifdef ENABLE_AMD_INSTRUCTION_SET
    return max3(x(t1), y(t1), z(t1));
#else
    return max(max(x(t1), y(t1)), z(t1));
#endif
}

float compmin3(in float t1){
#ifdef ENABLE_AMD_INSTRUCTION_SET
    return min3(x(t1), y(t1), z(t1));
#else
    return min(min(x(t1), y(t1)), z(t1));
#endif
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


