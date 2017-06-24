
vec4 swiz4(in vec4 vc){
    const uint sz = gl_SubGroupInvocationARB % 4;
    if (sz == 1) return vc.yyyy;
    if (sz == 2) return vc.zzzz;
    if (sz == 3) return vc.wwww;
    return vc.xxxx;
}

float swiz(in vec4 vc){
    const uint sz = gl_SubGroupInvocationARB % 4;
    if (sz == 1) return vc.y;
    if (sz == 2) return vc.z;
    if (sz == 3) return vc.w;
    return vc.x;
}

// defined lanes
float x(in float mem){
    return readInvocationARB(mem, (gl_SubGroupInvocationARB / 4) * 4);
}

float y(in float mem){
    return readInvocationARB(mem, (gl_SubGroupInvocationARB / 4) * 4 + 1);
}

float z(in float mem){
    return readInvocationARB(mem, (gl_SubGroupInvocationARB / 4) * 4 + 2);
}

float w(in float mem){
    return readInvocationARB(mem, (gl_SubGroupInvocationARB / 4) * 4 + 3);
}

// read lane from vector4
float lane(in float mem, in int l){
    return readInvocationARB(mem, (gl_SubGroupInvocationARB / 4) * 4 + (l % 4));
}

// compact vector
vec4 cvec4(in float mem){
    return vec4(x(mem), y(mem), z(mem), w(mem));
}

// matrix math on WARP%4 lanes (require compacted vector)
float mult4(in vec4 vec, in mat4 mat){
    return dot(mat[gl_SubGroupInvocationARB % 4], vec);
}

float mult4(in mat4 mat, in vec4 vec){
    return dot( vec4(swiz(mat[0]), swiz(mat[1]), swiz(mat[2]), swiz(mat[3])), vec );
}

// matrix math on WARP%4 lanes (laned version)
float mult4(in float vec, in mat4 mat){
    return mult4(cvec4(vec), mat);
}

float mult4(in mat4 mat, in float vec){
    return mult4(mat, cvec4(vec));
}

// is work lane (for most operations)
bool mt(){
    return (gl_SubGroupInvocationARB % 4) == 0;
}

// cross lane "cross product"
float cross(in float a, in float b){
    const uint ln = gl_SubGroupInvocationARB % 3;
    return dot(vec2(
        lane(a, (ln + 1) % 3),
        lane(b, (ln + 1) % 3)
    ), -vec2(
        lane(b, (ln + 2) % 3),
        lane(a, (ln + 2) % 3)
    ));
}