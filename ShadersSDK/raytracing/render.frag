#version 460 core

#define FRAGMENT_SHADER
#define SIMPLIFIED_RAY_MANAGMENT

#include "../include/constants.glsl"
#include "../include/structs.glsl"
#include "../include/uniforms.glsl"
#include "../include/rayslib.glsl"

#define NEIGHBOURS 8
#define AXES 4
#define POW2(a) ((a)*(a))
#define GEN_METRIC(before, center, after) POW2((center) * vec4(2.0f) - (before) - (after))
#define BAIL_CONDITION(new,original) (lessThanEqual(new, original))
#define SYMMETRY(a) (-a)
#define O(u,v) (ivec2(u, v))

layout ( location = 0 ) out vec4 outFragColor;
layout ( location = 0 ) in vec2 texcoord;
layout ( binding = 5 ) uniform sampler2D samples;

const ivec2 axes[AXES] = {O(-1, -1), O( 0, -1), O( 1, -1), O(-1,  0)};

vec4 filtered(in vec2 tx){
    ivec2 center_pix = ivec2(tx * textureSize(samples, 0));
    vec4 center_pix_cache = texelFetch(samples, center_pix, 0);
    vec4 metric_reference[AXES];

    for (int axis = 0; axis < AXES; axis++) {
        vec4 before_pix = texelFetch(samples, center_pix + axes[axis], 0);
        vec4 after_pix  = texelFetch(samples, center_pix + SYMMETRY(axes[axis]), 0);
        metric_reference[axis] = GEN_METRIC (before_pix, center_pix_cache, after_pix);
    }

    vec4 sum = center_pix_cache;
    vec4 cur = center_pix_cache;
    ivec4 count = ivec4(1);

    for (int y=-1;y<1;y++) {
        for (int x=-1;x<1;x++) {
            if (!(x == 0 && y == 0)) { // not center
                const ivec2 offset = O(x, y);
                vec4 pix   = texelFetch(samples, center_pix + offset, 0);
                vec4 value = (pix + cur) * (0.5f);
                ivec4 mask = {1, 1, 1, 0};
                for (int axis = 0; axis < AXES; axis++) {
                    vec4 before_pix = texelFetch(samples, center_pix + axes[axis], 0);
                    vec4 after_pix  = texelFetch(samples, center_pix + SYMMETRY(axes[axis]), 0);
                    vec4 metric_new = GEN_METRIC (before_pix, value, after_pix);
                    mask = ivec4(BAIL_CONDITION(metric_new, metric_reference[axis])) & mask;
                }
                sum   += mix(vec4(0.0f), value , bvec4(mask));
                count += mix(ivec4(0), ivec4(1), bvec4(mask));
            }
        }
    }

    return (sum/vec4(count));
}

void main() {
    vec3 color = filtered(texcoord).xyz;
    outFragColor = vec4(clamp(color.xyz, vec3(0.0f), vec3(1.0f)), 1.0f);
}
