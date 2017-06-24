
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

float x(inout float mem){
    return readInvocationARB((gl_SubGroupInvocationARB / 4) * 4);
}

float y(inout float mem){
    return readInvocationARB((gl_SubGroupInvocationARB / 4) * 4 + 1);
}

float z(inout float mem){
    return readInvocationARB((gl_SubGroupInvocationARB / 4) * 4 + 2);
}

float w(inout float mem){
    return readInvocationARB((gl_SubGroupInvocationARB / 4) * 4 + 3);
}

// compact vector
vec4 cvec4(inout float mem){
    return vec4(x(mem), y(mem), z(mem), w(mem));
}

// matrix math on WARP%4 lanes (require compacted vector)
float mult4(in vec4 vec, in mat4 mat){
    return dot(mat[gl_SubGroupInvocationARB % 4], vec);
}

float mult4(inout mat4 mat, in vec4 vec){
    return dot( vec4(swiz(mat[0]), swiz(mat[1]), swiz(mat[2]), swiz(mat[3])), vec );
}