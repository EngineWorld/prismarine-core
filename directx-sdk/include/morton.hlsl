#ifndef _MORTON_H
#define _MORTON_H

uint part1By2_64(in uint a){
    uint x = a;
    x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}


uint part1By5(in uint x){
    uint answer = 0;
    for (uint i = 0; i < 32 / 6; ++i) {
        answer |= ((x & (1u << i)) << 5u*i);
    }
    return answer;
}

uint encodeMorton3_64(in uint3 a, in uint3 b)
{
    return 
        (part1By5(a.x) << 0) | (part1By5(a.y) << 1) | (part1By5(a.z) << 2) | 
        (part1By5(b.x) << 3) | (part1By5(b.y) << 4) | (part1By5(b.z) << 5);
}




uint encodeMorton3_64(in uint3 a)
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

uint3 decodeMorton3_64(in uint code)
{
    return uint3(compact1By2_64(code >> 0), compact1By2_64(code >> 1), compact1By2_64(code >> 2));
}

#endif
