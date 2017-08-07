#ifndef _VERTEX_H
#define _VERTEX_H

layout ( std430, binding = 10 ) readonly buffer GeomMaterialsSSBO {int mats[];};

layout (binding = 0) uniform sampler2D vertex_texture;
layout (binding = 1) uniform sampler2D normal_texture;
layout (binding = 2) uniform sampler2D texcoords_texture;
layout (binding = 3) uniform sampler2D modifiers_texture;

const ivec2 mit[3] = {ivec2(0,0), ivec2(1,0), ivec2(0,1)};

ivec2 mosaicIdc(in ivec2 mosaicCoord, in int idc){
    return mosaicCoord + mit[idc];
}

ivec2 gatherMosaic(in ivec2 uniformCoord){
    return ivec2(uniformCoord.x * 3 + uniformCoord.y % 3, uniformCoord.y);
}

vec4 gatherMosaicCompDyn(in sampler2D vertices, in ivec2 mosaicCoord, const uint comp){
    if (comp == 0) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 0); else 
    if (comp == 1) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 1); else 
    if (comp == 2) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 2); else 
    if (comp == 3) return textureGather(vertices, (vec2(mosaicCoord) + 0.5f) / textureSize(vertices, 0), 3); else 
    return vec4(0.f);
}


vec4 fetchMosaic(in sampler2D vertices, in ivec2 mosaicCoord, in uint idc){
    return texelFetch(vertices, mosaicCoord + mit[idc], 0);
}

ivec2 getUniformCoord(in int indice){
    return ivec2(indice % 1023, indice / 1023);
}

ivec2 getUniformCoord(in uint indice){
    return ivec2(indice % 1023, indice / 1023);
}

#endif
