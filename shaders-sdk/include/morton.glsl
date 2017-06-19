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
    x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

uint encodeMorton3_64(in uvec3 a)
{
    return part1By2_64(a.x) | (part1By2_64(a.y) << 1) | (part1By2_64(a.z) << 2);
}

uint compact1By2_64(in uint a)
{
    uint x = a;
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

uvec3 decodeMorton3_64(in uint code)
{
    return uvec3(compact1By2_64(code >> 0), compact1By2_64(code >> 1), compact1By2_64(code >> 2));
}

#endif

#endif
