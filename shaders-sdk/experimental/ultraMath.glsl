// get relative invocation ID (active)
uint activeInvocation() {
    //const uint64_t activeBits = ballotARB(true) & gl_SubGroupLtMaskARB;
    //const uvec2 unpacked = unpackUint2x32(activeBits);
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
    const vec4 vc = readInvocationARB(_vc, offt4);
    //const uint sz = lane4;
    return ((sz == 1) ? vc.yyyy : ((sz == 2) ? vc.zzzz : ((sz == 3) ? vc.wwww : vc.xxxx)));
}


// get swizzle component from packed vectors
float swiz(in vec4 _vc) {
    const vec4 vc = readInvocationARB(_vc, offt4);
    //const uint sz = lane4;
    return ((sz == 1) ? vc.y : ((sz == 2) ? vc.z : ((sz == 3) ? vc.w : vc.x)));
}

float swiz(in vec3 _vc) {
    const vec3 vc = readInvocationARB(_vc, offt4);
    //const uint sz = lane4;
    return ((sz == 1) ? vc.y : ((sz == 2) ? vc.z : vc.x));
}

float swiz(in vec2 _vc) {
    const vec2 vc = readInvocationARB(_vc, offt4);
    //const uint sz = lane4;
    return ((sz == 1) ? vc.y : vc.x);
}

int swiz(in ivec2 _vc) {
    const ivec2 vc = readInvocationARB(_vc, offt4);
    //const uint sz = lane4;
    return ((sz == 1) ? vc.y : vc.x);
}

bool swiz(in bvec2 _vc) {
    const bvec2 vc = bvec2(readInvocationARB(uvec2(_vc), offt4));
    //const uint sz = lane4;
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
    const float _x = x(mem);
    const float _y = y(mem);
    return laneP == 1 ? _x : _y;
}

int swapXY(in int mem){
    const int _x = x(mem);
    const int _y = y(mem);
    return laneP == 1 ? _x : _y;
}

uint swapXY(in uint mem){
    const uint _x = x(mem);
    const uint _y = y(mem);
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


// get length of vector 3 (optimized)
float length3(in float a){
    return length(compvec3(a));
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
    return dot(mat[laneP&3], vec);
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


// cross lane "cross product"
float cross3(in float a, in float b){
    const uint ln = laneP&3;
    const uint ln1 = (ln+1)%3;
    const uint ln2 = (ln+2)%3;
    return dot(vec2(lane(a, ln1), lane(b, ln1)), vec2(lane(b, ln2), -lane(a, ln2)));
}


// normalize function for warp
float normalize3(in float a){
    return a / length3(a);
}


void putv3(in float inp, inout vec3 mem, inout int loc){
    if (loc == 0) mem.x = inp;
    if (loc == 1) mem.y = inp;
    if (loc == 2) mem.z = inp;
}

// if any valid
bool bs(in bool valid) {
    return ballotARB(valid && mt()) != 0;
}

// if all invalid
bool ibs(in bool valid) {
    return ballotARB(valid && mt()) == 0;
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


