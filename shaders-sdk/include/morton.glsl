#ifndef _MORTON_H
#define _MORTON_H

#ifdef INT64_MORTON

uint64_t part1By2_64(in uint a){
    uint64_t x = uint64_t(a);
    x &= 0x1fffffl;
    x = (x | (x << 32)) & 0x1f00000000ffffl;
    x = (x | (x << 16)) & 0x1f0000ff0000ffl;
    x = (x | (x << 8 )) & 0x100f00f00f00f00fl;
    x = (x | (x << 4 )) & 0x10c30c30c30c30c3l;
    x = (x | (x << 2 )) & 0x1249249249249249l;
    return x;
}

uint64_t encodeMorton3_64(in uvec3 a)
{
    return part1By2_64(a.x) | (part1By2_64(a.y) << 1) | (part1By2_64(a.z) << 2);
}

uint compact1By2_64(in uint64_t a)
{
    uint64_t x = a;
    x &= 0x1249249249249249l;
    x = (x | (x >>  2)) & 0x10c30c30c30c30c3l;
    x = (x | (x >>  4)) & 0x100f00f00f00f00fl;
    x = (x | (x >>  8)) & 0x1f0000ff0000ffl;
    x = (x | (x >> 16)) & 0x1f00000000ffffl;
    x = (x | (x >> 32)) & 0x1fffffl;
    return uint(x);
}

uvec3 decodeMorton3_64(in uint64_t code)
{
    return uvec3(compact1By2_64(code >> 0), compact1By2_64(code >> 1), compact1By2_64(code >> 2));
}

#else

uint part1By2_64(in uint a){
    uint x = a;
    x &= 0x3ff;
    x = (x | (x << 16)) & 0x30000ff;
    x = (x | (x << 8 )) & 0x300f00f;
    x = (x | (x << 4 )) & 0x30c30c3;
    x = (x | (x << 2 )) & 0x9249249;
    return x;
}

uint encodeMorton3_64(in uvec3 a)
{
    return part1By2_64(a.x) | (part1By2_64(a.y) << 1) | (part1By2_64(a.z) << 2);
}

uint compact1By2_64(in uint a)
{
    uint x = a;
    x &= 0x9249249;
    x = (x | (x >>  2)) & 0x30c30c3;
    x = (x | (x >>  4)) & 0x300f00f;
    x = (x | (x >>  8)) & 0x30000ff;
    x = (x | (x >> 16)) & 0x3ff;
    return x;
}

uvec3 decodeMorton3_64(in uint code)
{
    return uvec3(compact1By2_64(code >> 0), compact1By2_64(code >> 1), compact1By2_64(code >> 2));
}

#endif

#endif
