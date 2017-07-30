#ifndef _VERTEX_H
#define _VERTEX_H

RWStructuredBuffer<int> mats : register(u10);

Texture2D<float4> vertex_texture : register(t0);
Texture2D<float4> normal_texture : register(t1);
Texture2D<float4> texcoords_texture : register(t2);
Texture2D<float4> modifiers_texture : register(t3);

static const int2 mit[3] = {int2(0,0), int2(1,0), int2(0,1)};

int2 mosaicIdc(in int2 mosaicCoord, in int idc){
    return mosaicCoord + mit[idc];
}

int2 gatherMosaic(in int2 uniformCoord){
    return int2(uniformCoord.x * 3 + uniformCoord.y % 3, uniformCoord.y);
}

float4 fetchMosaic( Texture2D<float4> vertices, in int2 mosaicCoord, in uint idc){
    return vertices.Load(int3(mosaicCoord + mit[idc], 0));
}

int2 getUniformCoord(in int indice){
    return int2(indice % 1023, indice / 1023);
}

int2 getUniformCoord(in uint indice){
    return int2(indice % 1023, indice / 1023);
}

#endif
