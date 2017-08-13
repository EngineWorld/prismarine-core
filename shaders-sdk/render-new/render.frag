#version 460 core

#define FRAGMENT_SHADER

#include "../include/constants.glsl"
#include "../include/structs.glsl"
#include "../include/uniforms.glsl"
#include "../include/rays-new.glsl"

#define NEIGHBOURS 8
#define AXES       (NEIGHBOURS/2)
#define POW2(a) ((a)*(a))
#define GEN_METRIC(before, center, after) POW2((center) * vec4(2.0f) - (before) - (after))
#define BAIL_CONDITION(new,original) (lessThan(new, original))
#define SYMMETRY(a)  (NEIGHBOURS - (a) - 1)
#define O(u,v) (ivec2(u, v))

layout ( location = 0 ) out vec4 outFragColor;
layout ( location = 0 ) in vec2 texcoord;
layout ( binding = 5 ) uniform sampler2D samples;

void mediumSwap(inout vec4 c0, inout vec4 c1){
    vec4 mn = mix(c0, c1, lessThanEqual(c0, c1));
    vec4 mx = mix(c0, c1, greaterThanEqual(c0, c1));
    c0 = mn;
    c1 = mx;
}

vec4 median(in vec4 cl[3], in int n){
    for(int i=0;i<n-1;i++){
        for(int j=i+1;j<n;j++){
            mediumSwap(cl[i], cl[j]);
        }
    }
    return cl[n >> 1];
}

vec4 checkerFetch(in sampler2D samples, in ivec2 tx, in int lod){ 
    vec4 t00 = texelFetch(samples, tx, lod);
    if (RAY_BLOCK cameraUniform.interlace == 1){
        vec4 t10 = texelFetch(samples, tx + ivec2(1, 0), lod);
        vec4 t01 = texelFetch(samples, tx + ivec2(0, 1), lod);
        vec4 xyf[3] = {t10, t00, t01};
        t00 = median(xyf, 3) * 0.5f + t00 * 0.5f;
    }
    return t00;
}

const ivec2 offsets[NEIGHBOURS] = {
    O(-1, -1), O( 0, -1), O( 1, -1),
    O(-1,  0),            O( 1,  0),
    O(-1,  1), O( 0,  1), O( 1,  1)
};

vec4 filtered(in vec2 tx){
    ivec2 center_pix = ivec2(tx * textureSize(samples, 0));
    vec4 center_pix_cache = checkerFetch(samples, center_pix, 0);

    vec4 metric_reference[AXES];
    for (int axis = 0; axis < AXES; axis++) {
        vec4 before_pix = checkerFetch(samples, center_pix + offsets[axis], 0);
        vec4 after_pix  = checkerFetch(samples, center_pix + offsets[SYMMETRY(axis)], 0);
        metric_reference[axis] = GEN_METRIC (before_pix, center_pix_cache, after_pix);
    }

     vec4 sum = center_pix_cache;
     vec4 cur = center_pix_cache;
    ivec4 count = ivec4(1);

    for (int direction = 0; direction < NEIGHBOURS; direction++) {
         vec4 pix   = checkerFetch(samples, center_pix + offsets[direction], 0);
         vec4 value = (pix + cur) * (0.5f);
        ivec4 mask = {1, 1, 1, 0};
        for (int axis = 0; axis < AXES; axis++) {
            vec4 before_pix = checkerFetch(samples, center_pix + offsets[axis], 0);
            vec4 after_pix  = checkerFetch(samples, center_pix + offsets[SYMMETRY(axis)], 0);
            vec4 metric_new = GEN_METRIC (before_pix, value, after_pix);
            mask = ivec4(BAIL_CONDITION(metric_new, metric_reference[axis])) & mask;
        }
        sum   += mix(vec4(0.0f), value , bvec4(mask));
        count += mix(ivec4(0), ivec4(1), bvec4(mask));
    }

    return (sum/vec4(count));
}

void main() {
    vec3 color = filtered(texcoord).xyz;
    outFragColor = vec4(clamp(color.xyz, vec3(0.0f), vec3(1.0f)), 1.0f);
}
