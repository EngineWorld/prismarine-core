#ifndef _TEXTUREMANAGER_H
#define _TEXTUREMANAGER_H

struct TextureDescription {
#ifdef USE_EMULATED
    uvec2 offset;
#else
    sampler2D offset;
#endif
    uvec2 size;

    //Additional meta-data
    uint clamping_x;
    uint clamping_y;
    uint format;
    uint lower;

    uint stride;
    uint idx;
    uint internalStride;
    uint magFilter;
};

layout ( std430, binding=16 ) readonly buffer TextureDescriptionsSSBO {TextureDescription tdescs[];};
layout ( binding = 0 ) uniform usampler2DArray texelData;

const int REPEAT = 0;
const int CLAMP = 1;

// All formats
const int RGBA = 4;
const int RGB = 3;
const int GRAYSCALE_A = 2;
const int GRAYSCALE = 1;

const int RGBA8_ENCODING = 4; //i.e. uint8
const int RGBA16_ENCODING = 8; //i.e. uint16

// 8-bit masking
uint masks[6] = {
    0x00000000,
    0xFF000000,
    0xFFFF0000,
    0xFFFFFF00,
    0xFFFFFFFF,
    0xFFFFFFFF // Reserved
};

// uvec2 masking (16-bit)
uvec2 masks16[6] = {

    uvec2(0x00000000, 0x00000000),
    uvec2(0xFFFF0000, 0x00000000),
    uvec2(0xFFFFFFFF, 0x00000000),
    uvec2(0xFFFFFFFF, 0xFFFF0000),
    uvec2(0xFFFFFFFF, 0xFFFFFFFF),
    uvec2(0xFFFFFFFF, 0xFFFFFFFF) // Reserved
};

// 16-bit packing in uvec2
vec4 unpackUnorm4x16(in uvec2 v){
    return vec4(unpackUnorm2x16(v.x), unpackUnorm2x16(v.y));
}

uvec2 packUnorm4x16(in vec4 v){
    return uvec2(packUnorm2x16(v.xy), packUnorm2x16(v.zw));
}

#ifdef USE_INT64
vec4 unpackUnorm4x16_i64(in uint64_t v){
    return unpackUnorm4x16(unpackUint2x32(v));
}

uint64_t packUnorm4x16_i64(in vec4 v){
    return packUint2x32(packUnorm4x16(v));
}
#endif



const uint colorfill[4] = {0x00, 0x00, 0x00, 0xFF};

uint fetch8bit(in ivec2 texelc, in int xoffset, in uint stride, in uint from){
    return texelFetch( texelData, ivec3(texelc.x * stride + xoffset, texelc.y, from), 0).x /*& 0xFF*/; // too fat with bit mask...
}

int modi(in int a, in int b) {
    return ((a % b) + b) % b;
}

ivec2 modi(in ivec2 a, in ivec2 b) {
    return ((a % b) + b) % b;
}

// Emulator of texel fetching
vec4 texelFetchWrap(in TextureDescription desc, in vec2 norm, in ivec2 toff){
#ifndef HW_SAMPLER
    ivec2 coordinate = ivec2(desc.size.xy * norm) + toff;
    coordinate = mix(modi(coordinate, ivec2(desc.size.xy)), clamp(coordinate, ivec2(0), ivec2(desc.size.xy-1)), bvec2(desc.clamping_x == CLAMP, desc.clamping_y == CLAMP));
#endif

#ifdef USE_EMULATED
    uint internalStride = desc.internalStride;
    uint textureStride = desc.stride;
    uint texelRaw = 0;
    for (int i=0;i<4;i++) {
        if (i < textureStride) {
            texelRaw |= fetch8bit(coordinate, i, internalStride, desc.offset.x) << (i * 8);
        } else {
            texelRaw |= colorfill[i] << (i * 8);
        }
    }
    vec4 color = unpackUnorm4x8(texelRaw);
#else
#ifdef HW_SAMPLER
    vec4 color = textureOffset(desc.offset, norm, toff);
#else
    vec4 color = texelFetch(desc.offset, coordinate, 0);
#endif
#endif

    // In both result, reformat
    if (desc.format == GRAYSCALE_A) return color.rrrg;
    if (desc.format == GRAYSCALE  ) return vec4(color.rrr, 1.0f);
    if (desc.format == RGB ) return vec4(color.rgb, 1.0f);
    if (desc.format == RGBA) return vec4(color.rgba);
    return color;
}

vec4 texelFetchWrap(in TextureDescription desc, in vec2 norm){
#ifdef HW_SAMPLER
    return texture(desc.offset, norm);
#else
    return texelFetchWrap(desc, norm, ivec2(0));
#endif
}

vec4 texelFetchWrap(in uint binding, in vec2 norm){
    return texelFetchWrap(tdescs[binding], norm);
}

vec4 texelFetchWrap(in uint binding, in vec2 norm, in ivec2 toff){
    return texelFetchWrap(tdescs[binding], norm, toff);
}

vec4 texelFetchLinear(in TextureDescription desc, in vec2 norm, in ivec2 toff) {
#ifdef HW_SAMPLER
    return texelFetchWrap(desc, norm, toff);
#else
    norm -= 0.49999f / vec2(desc.size.xy);

    const vec2 tp = norm.xy;
    const vec4 a00 = texelFetchWrap(desc, tp, toff + ivec2(0, 0));
    const vec4 a10 = texelFetchWrap(desc, tp, toff + ivec2(1, 0));
    const vec4 a01 = texelFetchWrap(desc, tp, toff + ivec2(0, 1));
    const vec4 a11 = texelFetchWrap(desc, tp, toff + ivec2(1, 1));
    const vec2 coef = fract(norm * vec2(desc.size.xy));

    return mix(
        mix(a00, a10, coef.x),
        mix(a01, a11, coef.x),
        coef.y
    );
#endif
}

vec4 texelFetchLinear(in TextureDescription desc, in vec2 norm){
    return texelFetchLinear(desc, norm, ivec2(0));
}

vec4 texelFetchLinear(in uint binding, in vec2 norm){
    return texelFetchLinear(tdescs[binding], norm, ivec2(0));
}

vec4 texelFetchLinear(in uint binding, in vec2 norm, in ivec2 toff){
    return texelFetchLinear(tdescs[binding], norm, toff);
}

#endif
